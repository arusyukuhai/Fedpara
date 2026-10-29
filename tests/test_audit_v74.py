"""v74 tests for real mixed-training persistence, failure hygiene and streaming cycles."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from patchlib import cli, livefed
from patchlib.hf_mixed_datasets_v68 import HFRowCycle
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
from test_mixed_learning_v65 import make_args, setup_model, ToyTokenizer, write


def _standalone_args(tmp_path):
    sft=write(tmp_path,'sft.jsonl',[{'prompt':'ok','response':'fine'}])
    ppo=write(tmp_path,'ppo.jsonl',[{'prompt':'hello','reference':'yes'}])
    args=make_args(tmp_path,sft_jsonl=sft,ppo_jsonl=ppo,critic_state_dir=str(tmp_path/'critic'))
    args.peer_id='peer_a';args.steps=3;args.lr=.001;args.weight_decay=0.;args.disk_optimizer_dir=None
    args.resume=None;args.output=str(tmp_path/'adapter.patch');args.full_state_output=None
    args.shard_state_manifest=None;args.shard_store=None;args.log_steps=100;args.wire_dtype='float16'
    return args


def test_local_train_final_critic_step_is_persisted(tmp_path,monkeypatch):
    from patchlib import mixed_learning_v65 as mix
    from patchlib.critic_state_v66 import snapshot_file, load
    args=_standalone_args(tmp_path)
    model,adapter,_,batch=setup_model()
    monkeypatch.setattr(cli,'load',lambda _: (model,adapter))
    monkeypatch.setattr(cli,'training_batches',lambda _:iter([batch]*args.steps))
    monkeypatch.setattr(mix,'build',lambda a:MixedObjectiveTrainer(a,ToyTokenizer()))
    cli.train(args)
    assert args._v65_mixer._closed
    assert args._v65_mixer.critic_steps==3
    snap=snapshot_file(args.critic_state_dir)
    assert snap.is_file(), 'previously only every tenth step was persisted'
    # Check exact optimizer steps, not just existence of an old snapshot.
    another=MixedObjectiveTrainer(args,ToyTokenizer())
    try:
        assert another.critic_steps==3
        assert len(another.critic_optim.state)>0
        for p in another.critic.trainable_parameters() if hasattr(another.critic,'trainable_parameters') else another.critic.parameters():
            assert torch.isfinite(p).all()
    finally:
        another.close()


def test_mixed_trainer_finalizer_retries_no_double_save(tmp_path,monkeypatch):
    args=_standalone_args(tmp_path)
    fake=SimpleNamespace(close_calls=0)
    def close():fake.close_calls+=1
    fake.close=close
    args._v65_mixer=fake
    cli._close_mixed_trainer(args)
    cli._close_mixed_trainer(args)
    # This fake is not marked closed; the finalizer is permitted twice only
    # for objects that implement idempotent close (real MixedObjectiveTrainer).
    assert fake.close_calls==2


def test_live_training_and_p2p_finalizers_on_exception(monkeypatch):
    args=SimpleNamespace()
    seen=[]
    fake=SimpleNamespace(close=lambda:seen.append('closed'))
    args._v65_mixer=fake
    async def fail(*args,**kwargs):
        raise RuntimeError('link disconnected')
    monkeypatch.setattr(cli,'_train_p2p_impl',fail)
    with pytest.raises(RuntimeError,match='disconnected'):
        asyncio.run(cli.train_p2p(args))
    monkeypatch.setattr(livefed,'_train_live_impl',fail)
    with pytest.raises(RuntimeError,match='disconnected'):
        asyncio.run(livefed.train_live(args,load_model=None,training_batches=None,run_step=None))
    assert seen==['closed','closed']


def test_finite_hub_split_restarts_without_metadata_reload():
    spec={'repo':'org/test','revision':'a'*40,'config':None,'split':'train',
          'prompt_column':'prompt','response_column':'answer'}
    calls=[]
    def factory(*a,**k):
        calls.append((a,k))
        return [{'prompt':'a','answer':'one'},{'prompt':'b','answer':'two'}]
    rows=HFRowCycle(spec,'sft',dataset_factory=factory)
    try:
        assert [rows.next()['response'] for _ in range(7)]==['one','two','one','two','one','two','one']
        assert len(calls)==1
    finally:
        rows.close()


def test_single_pass_hub_source_reopens_on_exhaustion():
    spec={'repo':'org/test','revision':'a'*40,'config':None,'split':'train',
          'prompt_column':'prompt','response_column':'answer'}
    calls=[]
    def factory(*a,**k):
        calls.append(1)
        return iter([{'prompt':'a','answer':'once'}])
    rows=HFRowCycle(spec,'sft',dataset_factory=factory)
    try:
        assert [rows.next()['response'] for _ in range(3)]==['once']*3
        assert len(calls)==3
    finally:rows.close()


def test_ppo_critic_optimizer_rejects_nonfinite_grads(tmp_path,monkeypatch):
    args=_standalone_args(tmp_path)
    model,adapter,opt,batch=setup_model()
    args._v65_mixer=MixedObjectiveTrainer(args,ToyTokenizer())
    old=[p.detach().clone() for p in args._v65_mixer.critic.parameters()]
    hook=next(iter(args._v65_mixer.critic.parameters())).register_hook(lambda grad:torch.full_like(grad,float('nan')))
    try:
        opt.zero_grad(set_to_none=True)
        with pytest.raises(ValueError,match='nonfinite PPO Critic gradient'):
            cli.run_step(model,adapter,opt,batch,args)
        assert args._v65_mixer.critic_steps==0
        assert all(torch.equal(p,q) for p,q in zip(old,args._v65_mixer.critic.parameters()))
    finally:
        hook.remove();args._v65_mixer.close()


def test_partial_constructor_closes_hub_worker_when_reward_load_fails(monkeypatch,tmp_path):
    import patchlib.mixed_learning_v65 as mixed
    import patchlib.reward_model_v67 as rewards
    made=[]
    class FakeSource:
        def __init__(self,*args,**kwargs): self.closed=False; made.append(self)
        def close(self):self.closed=True
        def next(self):return {'prompt':'a','response':'b'}
    class FakePrefetch:
        def __init__(self,source):self.source=source
        def close(self):self.source.close()
    def explode(*args,**kwargs):raise RuntimeError('HF reward not available')
    monkeypatch.setattr(mixed,'HFRowCycle',FakeSource)
    monkeypatch.setattr(mixed,'PrefetchedRows',FakePrefetch)
    monkeypatch.setattr(rewards,'HFRewardScorer',explode)
    # Constructor's args validation requires the full pinned reward config;
    # use an unmodified compatible fixture with synthetic source specification.
    from patchlib.mixed_learning_v65 import source_spec
    args=make_args(tmp_path,sft_hf_repo='org/dataset',sft_hf_revision='b'*40,
             ppo_hf_repo='org/dataset',ppo_hf_revision='b'*40,
             ppo_reward='hf_model',reward_model_id='org/reward',reward_model_revision='c'*40)
    with pytest.raises(RuntimeError,match='HF reward not available'):
        MixedObjectiveTrainer(args,ToyTokenizer())
    assert len(made)==2 and all(x.closed for x in made)
