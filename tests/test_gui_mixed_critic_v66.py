"""Real WebView payload -> safe CLI flags, private paths, critic recovery."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import sys, types

import pytest
import torch
from patchlib.mix_gui_v66 import normalize, command_options, objective_policy, DEFAULT
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer, LocalValueHead, MixedLearningError
from patchlib.critic_state_v66 import save,load,CriticStateError,snapshot_file
from patchlib.roomid import create_room_id
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import FedparaBridge, compose_html


def file(tmp_path,name,row):
    f=tmp_path/name;f.write_text(json.dumps(row)+'\n',encoding='utf8');return f


def config(tmp_path,**kw):
    sft=file(tmp_path,'sft.jsonl',{'prompt':'Q','response':'A'})
    ppo=file(tmp_path,'ppo.jsonl',{'prompt':'Q','reference':'A'})
    return {**DEFAULT,'enabled':True,'sft_jsonl':str(sft),'ppo_jsonl':str(ppo),**kw}


def test_mixed_three_datasets_and_private_paths(tmp_path):
    opts=normalize(config(tmp_path,critic_kind='linear',critic_lr=.002),task='causal-lm')
    flags=command_options(opts,state_dir=tmp_path/'critic')
    assert flags[flags.index('--sft-jsonl')+1]==str(tmp_path/'sft.jsonl')
    assert flags[flags.index('--ppo-jsonl')+1]==str(tmp_path/'ppo.jsonl')
    assert flags[flags.index('--critic-kind')+1]=='linear'
    policy=objective_policy(opts)
    assert policy['version']=='pt-sft-clipped-ppo-v66'
    assert 'sft.jsonl' not in json.dumps(policy)
    assert 'ppo.jsonl' not in json.dumps(policy)
    assert 'critic' in policy and policy['critic']['kind']=='linear'


def test_sft_only_and_invalid_path(tmp_path):
    opts=config(tmp_path,ppo_jsonl='',ppo_weight=0)
    parsed=normalize(opts,task='causal-lm')
    assert parsed['ppo_jsonl']=='' and parsed['ppo_weight']==0
    bad=dict(opts,sft_jsonl=str(tmp_path/'unknown.jsonl'))
    with pytest.raises(ValueError,match='existing local'):
        normalize(bad,task='causal-lm')
    with pytest.raises(ValueError,match='causal'):
        normalize(opts,task='masked-lm')
    assert normalize({'enabled':False,'sft_jsonl':'/nonexistent'},task='causal-lm')==DEFAULT


def test_critic_checkpoint_roundtrip_and_wrong_identity(tmp_path):
    torch.manual_seed(9)
    model=LocalValueHead(kind='mlp',hidden=16)
    opt=torch.optim.AdamW(model.parameters(),lr=.005)
    x=torch.rand(5,4)
    for _ in range(3):
        opt.zero_grad();model(x).square().sum().backward();opt.step()
    expected=model(x).detach().clone()
    where=tmp_path/'critic'
    p=save(where,model,opt,kind='mlp',hidden=16,identity='room-123',steps=3)
    restored=LocalValueHead(kind='mlp',hidden=16)
    new_opt=torch.optim.AdamW(restored.parameters(),lr=.005)
    assert load(p,restored,new_opt,kind='mlp',hidden=16,identity='room-123')==3
    torch.testing.assert_close(expected,restored(x))
    for p1,p2 in zip(model.parameters(),restored.parameters()):
        assert opt.state[p1].keys()==new_opt.state[p2].keys()
        for k in opt.state[p1]:torch.testing.assert_close(opt.state[p1][k],new_opt.state[p2][k])
    with pytest.raises(CriticStateError,match='another room'):
        load(snapshot_file(where),restored,new_opt,kind='mlp',hidden=16,identity='other-room')
    with pytest.raises(CriticStateError,match='layout|kind'):
        load(snapshot_file(where),LocalValueHead(kind='linear',hidden=16),
             torch.optim.AdamW(LocalValueHead(kind='linear',hidden=16).parameters()),
             kind='linear',hidden=16,identity='room-123')


def make_controller(tmp_path,monkeypatch):
    c=WebviewController(tmp_path)
    room=create_room_id()
    c.save({'room':room,'model':'org/ourmodel','revision':'main',
            'task':'causal-lm','initialization':'pretrained','seed':42,
            'sources':[{'repo':'org/pretraining','revision':'main','config':None,
                        'split':'train','columns':{'text':'text'},'weight':1.0}]})
    called=[]
    monkeypatch.setattr(c.train,'start',lambda cmd,log:called.append(cmd) or 201)
    return c,room,called


def test_real_gui_to_cli_with_critic_and_pinned_policy(tmp_path,monkeypatch):
    c,room,called=make_controller(tmp_path,monkeypatch)
    b=FedparaBridge(c)
    settings=config(tmp_path,critic_kind='linear')
    payload={'room':room,'ws':'ws://127.0.0.1:8080/ws',
             'streaming_base':True,'kind':'loha','rank':16,
             'mixed_training':settings}
    started=b.start_training(payload)
    assert started['mixed_enabled'] is True and started['pid']==201
    cmd=called[-1]
    for x in ('--sft-jsonl','--ppo-jsonl','--critic-state-dir','--mix-policy'):
        assert x in cmd
    assert cmd[cmd.index('--pt-weight')+1]=='1.0'
    assert b.mixed_settings({'room':room})['critic_kind']=='linear'
    manifest=(tmp_path/'.fedpara-gui'/'rooms'/room/'manifest.json').read_text()
    assert str(tmp_path) not in manifest
    policyfile=Path(cmd[cmd.index('--mix-policy')+1]);policy=json.loads(policyfile.read_text())
    assert policy['objectives']==[True,True]
    assert str(tmp_path) not in policyfile.read_text()
    with pytest.raises(ValueError,match='immutable'):
        b.start_training({**payload,'mixed_training':{**settings,'ppo_clip':.1}})
    with pytest.raises(ValueError,match='immutable'):
        b.start_training({**payload,'mixed_training':{'enabled':False}})
    # Local dataset shards may change without changing the shared objective protocol.
    file(tmp_path,'sft2.jsonl',{'prompt':'b','response':'c'})
    b.start_training({**payload,'mixed_training':{**settings,
                      'sft_jsonl':str(tmp_path/'sft2.jsonl')}})


def test_existing_pt_only_journal_requires_new_room(tmp_path,monkeypatch):
    c,room,called=make_controller(tmp_path,monkeypatch)
    profile=c.profiles.get(room)
    # Simulate an established PT-only local journal.
    p=c._room_path(room).parent/'persistent-events.sqlite3'
    p.write_bytes(b'sqlite journal')
    monkeypatch.setattr('patchlib.webview_controller.allocate_profile',lambda folder: SimpleNamespace(name='test',journal=p))
    with pytest.raises(ValueError,match='NEW room'):
        c.start({'room':room,'ws':'ws://127.0.0.1:8080/ws','streaming_base':True,
                 'kind':'loha','rank':16,'mixed_training':config(tmp_path)})


def test_real_html_contains_mixed_controls_bridge():
    html=compose_html()
    for item in ('mixSftPath','mixPpoPath','mixCriticKind','mixCriticInit',
                 'mixPtDataset','mixCriticLr','mixed_training','mixed_settings'):
        assert item in html

class _ToyTokenizer:
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):
        return [3+(ord(c)%12) for c in text]


def test_critic_saved_during_real_ppo_updates_and_resumed(tmp_path):
    from patchlib.cli import run_step
    from patchlib.core import UniversalAdapter
    from torch import nn
    torch.manual_seed(123)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(18,12)
            self.lin=nn.Linear(12,12)
            self.head=nn.Linear(12,18)
        def forward(self,input_ids,attention_mask=None):
            return {'logits':self.head(torch.tanh(self.lin(self.emb(input_ids))))}
    model=Toy()
    adapter=UniversalAdapter(model,'toy','main',kind='lora',rank=2,target_scope='linear')
    optimizer=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.01)
    opts=config(tmp_path,critic_kind='linear',critic_hidden=8,critic_lr=.004,ppo_reuse=1)
    cfg=normalize(opts,task='causal-lm')
    args=SimpleNamespace(**cfg,room_id='test-room-critic',model='toy',max_length=32,
                         hf_task='causal-lm',loss_policy='auto',input_map='',
                         device='cpu',grad_accum=1,telemetry_file=None,batch_size=1,
                         critic_state_dir=tmp_path/'savedcritic')
    args._v65_mixer=MixedObjectiveTrainer(args,_ToyTokenizer())
    # run_step will keep the explicitly built Critic and train it alongside
    # actor LoRA. It must publish its 10th checkpoint without network I/O.
    batch={'input_ids':torch.tensor([[3,4,5,6,7]],dtype=torch.long),
           'attention_mask':torch.ones(1,5,dtype=torch.long),
           'labels':torch.tensor([[3,4,5,6,7]],dtype=torch.long)}
    for i in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss=run_step(model,adapter,optimizer,batch,args)
        assert isinstance(loss,float)
        optimizer.step()
    assert snapshot_file(args.critic_state_dir).is_file()
    saved_steps=args._v65_mixer.critic_steps
    before=[x.detach().clone() for x in args._v65_mixer.critic.parameters()]
    restarted=MixedObjectiveTrainer(args,_ToyTokenizer())
    assert restarted.critic_steps==saved_steps==10
    for prior,later in zip(before,restarted.critic.parameters()):
        torch.testing.assert_close(prior,later)
    restarted.close()
    args._v65_mixer.close()


def test_initial_external_critic_is_loaded_without_code_execution(tmp_path):
    from patchlib.critic_state_v66 import save
    c=LocalValueHead(kind='linear',hidden=32)
    opt=torch.optim.AdamW(c.parameters(),lr=.001)
    initial=save(tmp_path/'initial',c,opt,kind='linear',hidden=32,
                 identity='another-room',steps=0)
    p=file(tmp_path,'rollouts.jsonl',{'prompt':'start','reference':'goal'})
    args=SimpleNamespace(**{**DEFAULT,'enabled':True,'ppo_jsonl':str(p),
        'sft_weight':0,'critic_kind':'linear', 'critic_init':str(initial)},
        room_id='room2',max_length=32,hf_task='causal-lm',loss_policy='auto',
        input_map='',critic_state_dir=tmp_path/'local2')
    critic=MixedObjectiveTrainer(args,_ToyTokenizer())
    assert critic.critic is not None
    for a,b in zip(c.parameters(),critic.critic.parameters()):torch.testing.assert_close(a,b)
    critic.close()


def test_other_peers_can_use_distinct_objective_datasets(tmp_path, monkeypatch):
    """Room-wide objective semantics stay pinned; private dataset paths can differ."""
    c, room, called = make_controller(tmp_path, monkeypatch)
    root=config(tmp_path)
    base={'room':room,'ws':'ws://127.0.0.1:8080/ws',
          'streaming_base':True,'kind':'loha','rank':16}
    c.start({**base, 'mixed_training':root})
    # A second peer can be PT-only or have just one of the room's objectives.
    pt_only={**root,'sft_jsonl':'','ppo_jsonl':''}
    c.start({**base,'mixed_training':pt_only})
    assert '--mix-policy' in called[-1]
    assert '--sft-jsonl' not in called[-1]
    assert '--ppo-jsonl' not in called[-1]
    sft_only={**root,'ppo_jsonl':''}
    c.start({**base,'mixed_training':sft_only})
    assert '--sft-jsonl' in called[-1] and '--ppo-jsonl' not in called[-1]
    with pytest.raises(ValueError,match='immutable'):
        c.start({**base,'mixed_training':{**sft_only,'critic_hidden':64}})


def test_dataset_role_schema_validated_before_worker_starts(tmp_path):
    invalid=file(tmp_path,'bad.jsonl',{'prompt':'Hello','reference':'Yes'})
    with pytest.raises(ValueError,match='response'):
        normalize(config(tmp_path,sft_jsonl=str(invalid)),task='causal-lm')
    with pytest.raises(ValueError,match='reference'):
        normalize(config(tmp_path,ppo_jsonl=str(file(tmp_path,'no-reference.jsonl',
                  {'prompt':'Hi','response':'there'}))),task='causal-lm')


def test_gui_recovers_shared_policy_even_without_local_objective_files(tmp_path, monkeypatch):
    c,room,called=make_controller(tmp_path,monkeypatch)
    base={'room':room,'ws':'ws://127.0.0.1:8080/ws',
          'streaming_base':True,'kind':'loha','rank':16}
    c.start({**base,'mixed_training':config(tmp_path,critic_kind='linear',critic_lr=.002)})
    original=c.mixed_settings({'room':room})
    assert original['pinned_policy']['critic']['kind']=='linear'
    from patchlib.profiles import PeerProfile
    c.close()
    restarted=WebviewController(tmp_path)
    try:
        # No room-local paths are copied into the shared objective protocol.
        data=restarted.mixed_settings({'room':room})
        assert data['pinned_policy']['critic']['lr']==.002
        assert data['enabled'] is True
        assert data['sft_jsonl']==str(tmp_path/'sft.jsonl')
    finally:
        restarted.close()
