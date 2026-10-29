"""PT + masked SFT + on-policy PPO on the same signed adapter."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.cli import run_step
from patchlib.core import UniversalAdapter
from patchlib.mixed_learning_v65 import (
    MixedObjectiveTrainer, JsonlCycle, MixedLearningError, spec, validate_args,
)


class ToyTokenizer:
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):
        return [3+(ord(c)%12) for c in text]


class SmallCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(18,12)
        self.linear=nn.Linear(12,12)
        self.head=nn.Linear(12,18)
    def forward(self,input_ids,attention_mask=None):
        return {'logits':self.head(torch.tanh(self.linear(self.emb(input_ids))))}


def make_args(tmp_path,**changes):
    d=dict(sft_jsonl=None,ppo_jsonl=None,pt_weight=1.,sft_weight=1.,ppo_weight=.2,
           ppo_clip=.2,ppo_kl_coef=.02,ppo_value_coef=.5,ppo_gamma=.99,
           ppo_gae_lambda=.95,ppo_max_tokens=3,ppo_reuse=2,ppo_reward='token_match',
           max_length=24,hf_task='causal-lm',loss_policy='auto',device='cpu',
           grad_accum=1,telemetry_file=None,input_map='',batch_size=1)
    d.update(changes)
    return SimpleNamespace(**d)


def write(tmp_path,name,rows):
    p=tmp_path/name
    p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf8')
    return p


def setup_model():
    torch.manual_seed(2026)
    model=SmallCausalLM()
    adapter=UniversalAdapter(model,'toy','revision',kind='lora',rank=2,target_scope='linear')
    optimizer=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.03)
    batch={'input_ids':torch.tensor([[3,4,5,6,7]]),
           'attention_mask':torch.ones(1,5,dtype=torch.long),
           'labels':torch.tensor([[3,4,5,6,7]])}
    return model,adapter,optimizer,batch


def test_mixture_forward_backward_and_local_critic_only(tmp_path):
    sft=write(tmp_path,'sft.jsonl',[{'prompt':'Hi','response':'there'}])
    ppo=write(tmp_path,'ppo.jsonl',[{'prompt':'Hi','reference':'yes'}])
    args=make_args(tmp_path,sft_jsonl=sft,ppo_jsonl=ppo)
    model,adapter,opt,batch=setup_model()
    args._v65_mixer=MixedObjectiveTrainer(args,ToyTokenizer())
    before=[p.detach().clone() for p in adapter.trainable_parameters()]
    opt.zero_grad(set_to_none=True)
    value_before=[p.detach().clone() for p in args._v65_mixer.critic.parameters()]
    for _ in range(3):
        loss=run_step(model,adapter,opt,batch,args)
        assert torch.isfinite(torch.tensor(loss))
        opt.step();opt.zero_grad(set_to_none=True)
    assert args._v65_mixer.counts=={'pt':3,'sft':3,'ppo':3}
    assert any(not torch.equal(x,y) for x,y in zip(before,adapter.trainable_parameters()))
    assert any(not torch.equal(x,y) for x,y in zip(value_before,args._v65_mixer.critic.parameters()))
    assert args._v65_mixer.cached['old_lp'].requires_grad is False
    # Local critic must not be in signed adapter state.
    assert not any(id(a)==id(p) for a in args._v65_mixer.critic.parameters() for p in adapter.trainable_parameters())
    wire=adapter.export('peer',seq=1)
    assert wire.startswith(b'FPAT1')
    args._v65_mixer.close()


def test_sft_labels_ignore_prompt_and_preserve_response(tmp_path):
    sft=write(tmp_path,'sft.jsonl',[{'prompt':'abcdef','response':'xyz'}])
    args=make_args(tmp_path,sft_jsonl=sft,pt_weight=0.,max_length=12)
    model,adapter,opt,batch=setup_model()
    mixer=MixedObjectiveTrainer(args,ToyTokenizer())
    loss=mixer.sft_loss(model,torch.device('cpu'))
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in adapter.trainable_parameters())
    mixer.close()


def test_ppo_old_policy_is_frozen_clipping_and_reuse(tmp_path):
    ppo=write(tmp_path,'ppo.jsonl',[{'prompt':'ab','reference':'c'}])
    args=make_args(tmp_path,ppo_jsonl=ppo,ppo_reuse=2)
    model,adapter,opt,batch=setup_model()
    mixer=MixedObjectiveTrainer(args,ToyTokenizer())
    torch.manual_seed(50)
    a,critic,metric=mixer.ppo_loss(model,torch.device('cpu'))
    assert a.requires_grad and critic.requires_grad
    assert 0<=metric['reward']<=1
    assert abs(metric['kl'])<1e-5
    first=mixer.cached
    a.backward();critic.backward()
    opt.step();opt.zero_grad(set_to_none=True)
    a2,_,metric2=mixer.ppo_loss(model,torch.device('cpu'))
    assert mixer.cached is first and mixer.reuses==2
    assert metric2['kl']>=0
    mixer.ppo_loss(model,torch.device('cpu'))
    assert mixer.cached is not first and mixer.reuses==1
    mixer.close()


def test_mixture_genesis_does_not_contain_local_paths(tmp_path):
    args=make_args(tmp_path,sft_jsonl=tmp_path/'private/data.jsonl',ppo_jsonl=tmp_path/'private/rewards.jsonl')
    d=spec(args)
    assert 'private' not in json.dumps(d)
    assert d['reward_source']=='ground-truth-reference-v1'
    assert spec(make_args(tmp_path)) is None
    other=make_args(tmp_path,sft_jsonl=args.sft_jsonl,ppo_jsonl=args.ppo_jsonl,ppo_clip=.1)
    assert spec(other)!=spec(args)


def test_invalid_schema_truncation_and_args_fail_closed(tmp_path):
    p=write(tmp_path,'malformed.jsonl',[{'prompt':'a','response':'b'},
                                         {'prompt':'only one field'}])
    f=JsonlCycle(p)
    assert f.next()['prompt']=='a'
    assert f.next()['prompt']=='only one field'
    f.close()
    model,adapter,opt,batch=setup_model()
    p2=write(tmp_path,'bad.jsonl',[{'prompt':'a','reference':''}])
    args=make_args(tmp_path,ppo_jsonl=p2)
    trainer=MixedObjectiveTrainer(args,ToyTokenizer())
    with pytest.raises(MixedLearningError,match='nonempty'):
        trainer.ppo_loss(model,torch.device('cpu'))
    trainer.close()
    args2=make_args(tmp_path,sft_jsonl=p,ppo_clip=0)
    with pytest.raises(MixedLearningError,match='ppo_clip'):
        validate_args(args2)
    args3=make_args(tmp_path,sft_jsonl=p,hf_task='seq2seq-lm')
    with pytest.raises(MixedLearningError,match='causal-lm'):
        validate_args(args3)


def test_mixed_integration_does_not_change_legacy_loss(tmp_path):
    args=make_args(tmp_path)
    model,adapter,opt,batch=setup_model()
    loss=run_step(model,adapter,opt,batch,args)
    assert loss>0
    assert not hasattr(args,'_v65_mixer')


def test_shared_room_policy_permits_specialized_data_peers(tmp_path):
    sft=write(tmp_path,'sft.jsonl',[{'prompt':'x','response':'y'}])
    ppo=write(tmp_path,'ppo.jsonl',[{'prompt':'x','reference':'y'}])
    combined=make_args(tmp_path,sft_jsonl=sft,ppo_jsonl=ppo)
    shared=tmp_path/'mix_policy.json'
    shared.write_text(json.dumps(spec(combined)),encoding='utf8')
    ptonly=make_args(tmp_path,mix_policy=shared)
    sftonly=make_args(tmp_path,mix_policy=shared,sft_jsonl=sft,pt_weight=1.)
    ppoonly=make_args(tmp_path,mix_policy=shared,ppo_jsonl=ppo)
    assert spec(ptonly)==spec(sftonly)==spec(ppoonly)==spec(combined)
    validate_args(ptonly)
    shared_config=json.loads(shared.read_text());shared_config['weights'][1]=100
    shared.write_text(json.dumps(shared_config))
    with pytest.raises(MixedLearningError,match='disagree'):
        validate_args(ptonly)


def test_sft_only_can_skip_pretraining_dataset(tmp_path):
    from patchlib.cli import training_batches
    sft=write(tmp_path,'sft.jsonl',[{'prompt':'hi','response':'okay'}])
    args=make_args(tmp_path,pt_weight=0.,sft_jsonl=sft)
    args.batches_dir=args.text_file=args.jsonl=args.room_manifest=None
    assert next(training_batches(args))=={}
    model,adapter,opt,batch=setup_model()
    args._v65_mixer=MixedObjectiveTrainer(args,ToyTokenizer())
    assert run_step(model,adapter,opt,{},args)>0
    assert args._v65_mixer.counts=={'pt':0,'sft':1,'ppo':0}
    args._v65_mixer.close()


def test_actual_local_train_command_exports_signed_adapter_with_all_three_losses(tmp_path,monkeypatch):
    from patchlib import cli
    from patchlib import mixed_learning_v65 as mix
    sft=write(tmp_path,'supervision.jsonl',[{'prompt':'How? ','response':'Yes.'}])
    ppo=write(tmp_path,'verifier.jsonl',[{'prompt':'Say ','reference':'yes'}])
    model,adapter,_,batch=setup_model()
    args=make_args(tmp_path,sft_jsonl=sft,ppo_jsonl=ppo)
    args.peer_id='peer_test';args.steps=3;args.lr=.002;args.weight_decay=0.
    args.disk_optimizer_dir=None;args.resume=None;args.output=str(tmp_path/'local.fpatch')
    args.full_state_output=None;args.shard_state_manifest=None;args.shard_store=None
    args.log_steps=1;args.wire_dtype='float16'
    monkeypatch.setattr(cli,'load',lambda unused:(model,adapter))
    monkeypatch.setattr(cli,'training_batches',lambda unused:iter([batch]*args.steps))
    monkeypatch.setattr(mix,'build',lambda unused:MixedObjectiveTrainer(unused,ToyTokenizer()))
    cli.train(args)
    assert (tmp_path/'local.fpatch').read_bytes().startswith(b'FPAT1')
    assert args._v65_mixer.counts=={'pt':3,'sft':3,'ppo':3}
    args._v65_mixer.close()


def test_ppo_refuses_unsupervised_self_grading(tmp_path):
    data=write(tmp_path,'bad_ppo.jsonl',[{'prompt':'Any question','reward':100}])
    args=make_args(tmp_path,ppo_jsonl=data)
    model,adapter,opt,batch=setup_model()
    trainer=MixedObjectiveTrainer(args,ToyTokenizer())
    with pytest.raises(MixedLearningError,match='reference strings'):
        trainer.ppo_loss(model,torch.device('cpu'))
    trainer.close()
