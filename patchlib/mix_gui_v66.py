"""Validate local PT/SFT/PPO and value critic settings before launching any peer.

Data paths stay local; the immutable objective protocol intentionally excludes
private training paths, while the pinned policy prevents unsafe silent joins.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import secrets

FIELDS=('critic_offpolicy','critic_rollout_gossip','distributed_critic', 'sft_hf_revision', 'sft_hf_repo', 'enabled', 'sft_jsonl', 'ppo_jsonl', 'pt_weight', 'sft_weight', 'ppo_weight', 'ppo_clip', 'ppo_kl_coef', 'ppo_value_coef', 'ppo_gamma', 'ppo_gae_lambda', 'ppo_max_tokens', 'ppo_reuse', 'ppo_reward', 'critic_kind', 'critic_hidden', 'critic_lr', 'critic_init', 'critic_hf_repo', 'critic_hf_revision', 'critic_hf_loader', 'critic_hf_device', 'critic_hf_max_length', 'reward_model_id', 'reward_model_revision', 'reward_model_device', 'reward_model_idle_offload', 'reward_model_loader', 'reward_model_output', 'reward_model_format', 'reward_model_index', 'reward_model_scale', 'reward_model_max_length', 'sft_hf_config', 'sft_hf_split', 'sft_hf_prompt_column', 'sft_hf_response_column', 'ppo_hf_repo', 'ppo_hf_revision', 'ppo_hf_config', 'ppo_hf_split', 'ppo_hf_prompt_column', 'ppo_hf_reference_column')
DEFAULT={'critic_offpolicy':'auto','critic_rollout_gossip': False,'distributed_critic': False, 'sft_hf_repo': '', 'sft_hf_revision': '', 'sft_hf_config': '', 'sft_hf_split': 'train', 'sft_hf_prompt_column': 'prompt', 'sft_hf_response_column': 'response', 'ppo_hf_repo': '', 'ppo_hf_revision': '', 'ppo_hf_config': '', 'ppo_hf_split': 'train', 'ppo_hf_prompt_column': 'prompt', 'ppo_hf_reference_column': 'reference', 'enabled': False, 'sft_jsonl': '', 'ppo_jsonl': '', 'pt_weight': 1.0, 'sft_weight': 1.0, 'ppo_weight': 0.2, 'ppo_clip': 0.2, 'ppo_kl_coef': 0.02, 'ppo_value_coef': 0.5, 'ppo_gamma': 0.99, 'ppo_gae_lambda': 0.95, 'ppo_max_tokens': 8, 'ppo_reuse': 2, 'ppo_reward': 'token_match', 'critic_kind': 'mlp', 'critic_hidden': 32, 'critic_lr': 0.001, 'critic_init': '', 'critic_hf_repo': '', 'critic_hf_revision': '', 'critic_hf_loader': 'sequence_classification', 'critic_hf_device': 'cpu', 'critic_hf_max_length': 512, 'reward_model_id': '', 'reward_model_revision': '', 'reward_model_device': 'cpu', 'reward_model_idle_offload': False, 'reward_model_loader': 'sequence_classification', 'reward_model_output': 'logits', 'reward_model_format': 'plain', 'reward_model_index': 0, 'reward_model_scale': 1.0, 'reward_model_max_length': 512}


def _path(value, *, ext, role=None, hf_reward=False):
    if not isinstance(value,str) or len(value)>4096 or '\x00' in value:
        raise ValueError('Expected a local dataset/critic file path')
    if not value.strip():return ''
    path=Path(value).expanduser().resolve()
    if path.suffix.lower()!=ext or not path.is_file():
        raise ValueError(f'Expected an existing local {ext} file: {path}')
    size=path.stat().st_size
    if not (0<size<=4*1024**3 if ext=='.jsonl' else 0<size<=16*1024**2):
        raise ValueError('Training file is empty or exceeds allowed size')
    if ext=='.jsonl':
        with path.open('r',encoding='utf8') as f:
            for line in f:
                if line.strip():
                    if len(line)>65536:raise ValueError('Dataset JSONL line exceeds 64KiB')
                    obj=json.loads(line)
                    if not isinstance(obj,dict):raise ValueError('Dataset row must be a JSON object')
                    required=('prompt','response') if role=='sft' else (('prompt',) if hf_reward else ('prompt','reference')) if role=='ppo' else ()
                    for name in required:
                        if not isinstance(obj.get(name),str) or not obj[name] or len(obj[name])>16384:
                            raise ValueError(f'{role} dataset must contain nonempty {name} text <=16Ki chars')
                    break
            else:raise ValueError('Dataset has no JSONL records')
    return str(path)


def normalize(payload, *, task, allow_pt_only=False):
    if not isinstance(payload,dict):raise ValueError('Training configuration must be an object')
    unknown=set(payload)-set(FIELDS)
    if unknown:raise ValueError('Unknown mixed-training options: '+str(sorted(unknown)))
    values={**DEFAULT,**payload}
    if type(values['enabled']) is not bool:raise ValueError('Mixed training enabled must be boolean')
    if not values['enabled']:
        # Never preserve stale dataset settings in a PT-only launch.
        return dict(DEFAULT)
    if task!='causal-lm':raise ValueError('Simultaneous PT/SFT/PPO currently needs a causal LM')
    for key in ('sft_jsonl','ppo_jsonl'):
        values[key]=_path(values[key],ext='.jsonl',role=key.removesuffix('_jsonl'),
                          hf_reward=(key=='ppo_jsonl' and values['ppo_reward']=='hf_model'))
    from .hf_mixed_datasets_v68 import source_spec, source_selected
    source_spec(SimpleNamespace(**values),'sft');source_spec(SimpleNamespace(**values),'ppo')
    values['critic_init']=_path(values['critic_init'],ext='.safetensors')
    if values['critic_kind']=='hf':
        from .hf_value_critic_v69 import critic_spec
        critic_spec(SimpleNamespace(**values))
        if values['critic_init']:raise ValueError('Hub Critic loads pretrained weights, not a local Critic initial file')
    for key in ('pt_weight','sft_weight','ppo_weight','ppo_clip','ppo_kl_coef',
                'ppo_value_coef','ppo_gamma','ppo_gae_lambda','critic_lr','reward_model_scale'):
        raw=values[key]
        if type(raw) not in (int,float) or not math.isfinite(float(raw)):
            raise ValueError(key+' must be a finite number')
        values[key]=float(raw)
    for key in ('critic_hidden','critic_hf_max_length','ppo_max_tokens','ppo_reuse','reward_model_index','reward_model_max_length'):
        if type(values[key]) is not int:raise ValueError(key+' must be an integer')
    if values['critic_kind'] not in ('mlp','linear','hf') or values['ppo_reward'] not in ('token_match','exact','hf_model'):
        raise ValueError('Unsupported critic or PPO reward type')
    if type(values['reward_model_idle_offload']) is not bool:
        raise ValueError('Reward idle offload must be a boolean')
    if type(values['distributed_critic']) is not bool:raise ValueError('distributed_critic must be boolean')
    if type(values['critic_rollout_gossip']) is not bool:raise ValueError('critic_rollout_gossip must be boolean')
    if values['critic_offpolicy'] not in ('auto','gae','trajectory_is'):
        raise ValueError('Critic off-policy correction must be auto, gae or trajectory_is')
    if values['ppo_reward']=='hf_model':
        from .reward_model_v67 import reward_spec
        reward_spec(SimpleNamespace(**values))
        if values['reward_model_device'] not in ('cpu','cuda','mps'):
            raise ValueError('Invalid Reward Model inference device')
    if not (source_selected(SimpleNamespace(**values),'sft') or source_selected(SimpleNamespace(**values),'ppo')):
        if not allow_pt_only or values['pt_weight']<=0:
            raise ValueError('Choose an SFT/PPO JSONL, or join an existing pinned mixed room as a PT-only peer')
    if values['ppo_weight']>0 and not source_selected(SimpleNamespace(**values),'ppo') and not allow_pt_only:
        raise ValueError('PPO weight is nonzero: choose a PPO dataset')
    if values['sft_weight']>0 and not source_selected(SimpleNamespace(**values),'sft') and not allow_pt_only:
        raise ValueError('SFT weight is nonzero: choose an SFT dataset')
    from .mixed_learning_v65 import validate_args
    args=SimpleNamespace(**{k:values[k] for k in values},
                         hf_task=task,loss_policy='auto',max_length=128,
                         input_map='',mix_policy=None)
    validate_args(args)
    return values


def command_options(values, *, state_dir):
    if not values['enabled']:return []
    result=['--pt-weight',str(values['pt_weight']),
            '--sft-weight',str(values['sft_weight']),
            '--ppo-weight',str(values['ppo_weight']),
            '--ppo-clip',str(values['ppo_clip']),
            '--ppo-kl-coef',str(values['ppo_kl_coef']),
            '--ppo-value-coef',str(values['ppo_value_coef']),
            '--ppo-gamma',str(values['ppo_gamma']),
            '--ppo-gae-lambda',str(values['ppo_gae_lambda']),
            '--ppo-max-tokens',str(values['ppo_max_tokens']),
            '--ppo-reuse',str(values['ppo_reuse']),
            '--ppo-reward',values['ppo_reward'],
            '--critic-kind',values['critic_kind'],
            '--critic-hidden',str(values['critic_hidden']),
            '--critic-lr',str(values['critic_lr']),
            '--critic-state-dir',str(state_dir)]
    if values['distributed_critic']:result.append('--distributed-critic')
    if values['critic_rollout_gossip']:result.append('--critic-rollout-gossip')
    result.extend(['--critic-offpolicy',values['critic_offpolicy']])
    if values['critic_kind']=='hf':
        for key,flag in (('critic_hf_repo','--critic-hf-repo'),
                         ('critic_hf_revision','--critic-hf-revision'),
                         ('critic_hf_loader','--critic-hf-loader'),
                         ('critic_hf_device','--critic-hf-device'),
                         ('critic_hf_max_length','--critic-hf-max-length')):
            result.extend([flag,str(values[key])])
    if values['ppo_reward']=='hf_model':
        for key,flag in (('reward_model_id','--reward-model-id'),
                ('reward_model_revision','--reward-model-revision'),
                ('reward_model_device','--reward-model-device'),
                ('reward_model_loader','--reward-model-loader'),
                ('reward_model_output','--reward-model-output'),
                ('reward_model_format','--reward-model-format'),
                ('reward_model_index','--reward-model-index'),
                ('reward_model_scale','--reward-model-scale'),
                ('reward_model_max_length','--reward-model-max-length')):
            result.extend([flag,str(values[key])])
        if values['reward_model_idle_offload']:
            result.append('--reward-model-idle-offload')
    for role in ('sft','ppo'):
        if values.get(role+'_hf_repo'):
            for suffix in ('repo','revision','config','split','prompt_column',('response' if role=='sft' else 'reference')+'_column'):
                if values[role+'_hf_'+suffix]:
                    result.extend(['--'+role+'-hf-'+suffix.replace('_','-'),values[role+'_hf_'+suffix]])
    for key,flag in (('sft_jsonl','--sft-jsonl'),('ppo_jsonl','--ppo-jsonl'),
                     ('critic_init','--critic-init')):
        if values[key]:result.extend([flag,values[key]])
    return result


def objective_policy(values, *, pinned=None):
    if not values['enabled']:return None
    from .mixed_learning_v65 import spec
    args=SimpleNamespace(**values,input_map='',mix_policy=pinned)
    return spec(args)


def atomic_json(path,document):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    tmp=path.parent/('.mix-'+secrets.token_hex(10))
    try:
        fd=os.open(tmp,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(document,f,sort_keys=True,separators=(',',':'))
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        d=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:tmp.unlink(missing_ok=True)
