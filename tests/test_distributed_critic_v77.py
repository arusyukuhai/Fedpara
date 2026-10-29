"""Signed distributed value-head convergence/replay/Byzantine-input tests."""
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from patchlib.core import PatchError
from patchlib.distributed_critic_v77 import CriticGossip, pack, unpack, critic_params
from patchlib.mixed_learning_v65 import LocalValueHead


def peer(tmp_path, name, genesis, peers=('alpha','beta','gamma'), kind='linear'):
    torch.manual_seed(177)
    critic=LocalValueHead(kind,32)
    mixer=SimpleNamespace(critic=critic,critic_optim=torch.optim.AdamW(critic.parameters(),lr=.01),
        args=SimpleNamespace(critic_kind=kind,critic_hidden=32),
        _critic_identity='shared-room-id',cached=None,reuses=0)
    def sign(n,d):return 'sig:'+d
    def verify(info):
        if info['sig']!='sig:'+info['sha256']:raise PatchError('bad signature')
    lane=CriticGossip(tmp_path/(name+'.sqlite3'),genesis,mixer,name,signer=sign,
                     trusted={p:verify for p in peers})
    return lane


def weights(lane):return [p.detach().clone() for p in lane.params]

def update(lane, amount=.01):
    before=weights(lane)
    with torch.no_grad():
        for p in lane.params:p.add_(amount)
    return lane.record_local(before,actor_tip=[0,None])


def test_signed_critic_two_peer_gossip_and_replay(tmp_path):
    a=peer(tmp_path,'alpha','a'*64);b=peer(tmp_path,'beta','a'*64)
    try:
        ab=update(a,.01)
        assert b.ingest(ab)
        assert not b.ingest(ab)
        assert [torch.equal(x,y) for x,y in zip(weights(a),weights(b))]==[True]*len(a.params)
        ba=update(b,.02)
        assert a.ingest(ba)
        for x,y in zip(weights(a),weights(b)):assert torch.allclose(x,y,atol=1e-6)
        expected=weights(a)
    finally:
        a.close();b.close()
    a2=peer(tmp_path,'alpha','a'*64)
    try:
        assert a2.journal.count()==2
        for x,y in zip(weights(a2),expected):assert torch.allclose(x,y,atol=1e-6)
    finally:a2.close()


def test_late_join_gap_repair_and_duplicate(tmp_path):
    a=peer(tmp_path,'alpha','b'*64);b=peer(tmp_path,'beta','b'*64)
    try:
        wires=[update(a,.001) for _ in range(10)]
        assert b.ingest(wires[-1])
        assert b.need(a.journal.frontiers())[:3]==[('alpha',1),('alpha',2),('alpha',3)]
        for ident in b.need(a.journal.frontiers(),limit=16):
            assert b.ingest(a.journal.get(*ident))
        assert not b.need(a.journal.frontiers())
        for x,y in zip(weights(a),weights(b)):
            assert torch.allclose(x,y,atol=1e-6)
    finally:a.close();b.close()


def test_tamper_wrong_identity_and_wrong_base(tmp_path):
    a=peer(tmp_path,'alpha','c'*64);b=peer(tmp_path,'beta','c'*64)
    try:
        wire=update(a,.001)
        before=weights(b)
        bad=bytearray(wire);bad[-1]^=1
        with pytest.raises(PatchError):b.ingest(bytes(bad))
        for x,y in zip(before,weights(b)):assert torch.equal(x,y)
        with pytest.raises(PatchError):unpack(wire,'d'*64,b.params,b.trusted)
        with pytest.raises(PatchError):unpack(wire,b.genesis,b.params,{'alpha':lambda info:(_ for _ in ()).throw(PatchError('not trusted'))})
    finally:a.close();b.close()
    # Different deterministic initial weights: cannot accidentally join same lane.
    torch.manual_seed(999)
    actor=LocalValueHead('linear',32)
    mixer=SimpleNamespace(critic=actor,critic_optim=torch.optim.AdamW(actor.parameters()),
        args=SimpleNamespace(critic_kind='linear',critic_hidden=32),
        _critic_identity='shared-room-id',cached=None,reuses=0)
    with pytest.raises((PatchError,ValueError)):
        lane=CriticGossip(tmp_path/'alpha.sqlite3','c'*64,mixer,'alpha',signer=lambda n,d:'sig:'+d)
        lane.close()


def test_foreign_update_resets_stale_moments_and_rollout(tmp_path):
    a=peer(tmp_path,'alpha','d'*64);b=peer(tmp_path,'beta','d'*64)
    try:
        b.mixer.critic_optim.state[b.params[0]]={'step':torch.tensor(1.)}
        b.mixer.cached={'stale':1};b.mixer.reuses=3
        assert b.ingest(update(a,.01))
        assert not b.mixer.critic_optim.state
        assert b.mixer.cached is None and b.mixer.reuses==0
    finally:a.close();b.close()


def test_nonfinite_overlarge_and_layout_rejected(tmp_path):
    a=peer(tmp_path,'alpha','e'*64)
    try:
        n=len(a.params)
        zeros=[torch.zeros_like(p) for p in a.params]
        with pytest.raises(PatchError):pack('alpha',1,a.genesis,[torch.full_like(x,float('nan')) for x in zeros],a.signer)
        with pytest.raises(PatchError):a.ingest(pack('alpha',1,a.genesis,[torch.full_like(x,11) for x in zeros],a.signer))
        wire=pack('alpha',1,a.genesis,[torch.zeros(2) for _ in range(n)],a.signer)
        with pytest.raises(PatchError):a.ingest(wire)
    finally:a.close()


def test_critics_are_protocol_pinned_and_local_paths_not_shared(tmp_path):
    from patchlib.mixed_learning_v65 import spec, validate_args, MixedLearningError
    ds=tmp_path/'examples.jsonl'
    ds.write_text('{"prompt":"Q","reference":"A"}\n')
    opts=SimpleNamespace(ppo_jsonl=ds,sft_jsonl=None,distributed_critic=True,critic_offpolicy='gae',
        ppo_weight=.2,pt_weight=1.,sft_weight=1.,hf_task='causal-lm',
        loss_policy='auto',ppo_clip=.2,ppo_kl_coef=.02,ppo_value_coef=.5,
        ppo_gamma=.99,ppo_gae_lambda=.95,ppo_max_tokens=8,ppo_reuse=2,
        max_length=128,critic_kind='linear',critic_hidden=32,critic_lr=.001,
        ppo_reward='token_match')
    validate_args(opts)
    policy=spec(opts)
    assert policy['version']=='pt-sft-clipped-ppo-v78'
    assert policy['critic_distribution']=='signed-root-critic-gossip-v78'
    assert str(ds) not in str(policy)
    opts.distributed_critic=False
    assert spec(opts)!=policy
    opts.distributed_critic=True;opts.critic_init=tmp_path/'local.safetensors'
    with pytest.raises(MixedLearningError,match='common pinned'):
        validate_args(opts)


def test_real_ppo_step_seals_critic_update_and_remote_receives(tmp_path):
    from patchlib.cli import run_step
    from patchlib.core import UniversalAdapter
    from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
    from torch import nn
    class ToyTokenizer:
        eos_token_id=2
        def encode(self,text,add_special_tokens=False):return [3+(ord(c)%12) for c in text]
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(18,12)
            self.lin=nn.Linear(12,12)
            self.head=nn.Linear(12,18)
        def forward(self,input_ids,attention_mask=None):
            return {'logits':self.head(torch.tanh(self.lin(self.emb(input_ids))))}
    ds=tmp_path/'ppo.jsonl';ds.write_text('{"prompt":"hi","reference":"ok"}\n')
    args=SimpleNamespace(ppo_jsonl=ds,sft_jsonl=None,distributed_critic=True,mode='elastic',
        ppo_weight=.2,pt_weight=0.,sft_weight=0.,ppo_clip=.2,
        ppo_kl_coef=.02,ppo_value_coef=.5,ppo_gamma=.99,ppo_gae_lambda=.95,
        ppo_max_tokens=3,ppo_reuse=2,max_length=32,critic_kind='linear',
        critic_hidden=32,critic_lr=.001,ppo_reward='token_match',
        hf_task='causal-lm',loss_policy='auto',device='cpu',grad_accum=1,
        telemetry_file=None,input_map='',batch_size=1)
    torch.manual_seed(24)
    model=Tiny();adapter=UniversalAdapter(model,'toy','head',kind='lora',rank=2,target_scope='linear')
    optim=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.003)
    mixer=MixedObjectiveTrainer(args,ToyTokenizer())
    args._v65_mixer=mixer;args._v65_mixer_adapter=adapter
    gen='f'*64
    def signer(n,d):return 'signed:'+d
    def verifier(info):
        if info['sig']!='signed:'+info['sha256']:raise PatchError('forged')
    def fresh_mixer():
        return MixedObjectiveTrainer(args,ToyTokenizer())
    first=[p.detach().clone() for p in critic_params(mixer)]
    a=CriticGossip(tmp_path/'a.sqlite3',gen,mixer,'alpha',signer=signer,
                   trusted={'alpha':verifier,'beta':verifier})
    other=fresh_mixer()
    # Same initial model for a separate PPO peer's value head.
    with torch.no_grad():
        for p,original in zip(critic_params(other),first):p.copy_(original)
    b=CriticGossip(tmp_path/'b.sqlite3',gen,other,'beta',signer=signer,
                   trusted={'alpha':verifier,'beta':verifier})
    try:
        result=run_step(model,adapter,optim,{},args)
        assert torch.isfinite(torch.tensor(result))
        assert len(args._v77_critic_before)==len(a.params)
        wire=a.record_local(args._v77_critic_before)
        assert b.ingest(wire)
        for x,y in zip(a.params,b.params):
            torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    finally:
        a.close();b.close();mixer.close();other.close()
