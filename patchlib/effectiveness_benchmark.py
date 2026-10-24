"""Offline causal-LM effectiveness study using Fedpara's actual UniversalAdapter.

NOT a network test. Peer updates run serially in this process; dividing their
compute seconds by the number of peers is a counterfactual *ideal parallel*
compute metric, not observed distributed wall time. Both centralized and peers retain LoHa/AdamW moments for the whole run.
Per-peer deltas use before/after effective LoHa weights, exactly as in live
continual elastic v2. The benchmark's all-peer barrier is idealized; live
P2P gossip may have stale bases and differing denominators.
"""
from __future__ import annotations

import csv
import copy
from functools import lru_cache
import itertools
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .core import UniversalAdapter

# Pinned official TinyStories-1M GPT-Neo checkpoint (2025-12-18 commit).
# The repository provides pytorch_model.bin, NOT safetensors. Only load with
# Transformers weights_only=True and a recent PyTorch secure unpickler.
TINYSTORIES_1M_ID = "roneneldan/TinyStories-1M"
TINYSTORIES_1M_REVISION = "77f1b168e219585646439073245fe87e56b3023e"
TINYSTORIES_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


class TinyCausalLM(nn.Module):
    """Small, genuine autoregressive self-attention LM; no external downloads."""
    def __init__(self, vocab: int = 32, width: int = 24, context: int = 12):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.norm1 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, width * 2)
        self.ff2 = nn.Linear(width * 2, width)
        self.norm2 = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.context = context

    def forward(self, x):
        batch, length = x.shape
        h = self.token(x) + self.position(torch.arange(length, device=x.device)).unsqueeze(0)
        q, k, v = self.query(h), self.key(h), self.value(h)
        weights = q @ k.transpose(-1,-2) / math.sqrt(q.shape[-1])
        forbidden = torch.ones((length,length),dtype=torch.bool,device=x.device).triu(1)
        weights = weights.masked_fill(forbidden, torch.finfo(weights.dtype).min)
        h = self.norm1(h + self.mix(torch.softmax(weights,dim=-1) @ v))
        return self.head(self.norm2(h + self.ff2(F.gelu(self.ff1(h)))))


def _data(peers: int, sequences: int, context: int, vocab: int, seed: int, validation_sequences: int):
    """Heldout is disjoint, and each peer has a slightly different chain."""
    generator = torch.Generator(device='cpu').manual_seed(seed + 41)
    # Shared signal plus local distributions: data is non-IID across peers.
    # Next token is usually current+1, sometimes current+peer_offset.
    def sample(peer: int, size: int):
        x = torch.randint(0, vocab, (size,context+1), generator=generator)
        for t in range(context):
            noise = torch.rand(size,generator=generator)
            step = torch.where(noise < .75, 1,
                   torch.where(noise < .88, (peer % 3) + 2, 0))
            x[:,t+1] = (x[:,t] + step) % vocab
        return x
    train = [sample(i,sequences) for i in range(peers)]
    counts = [validation_sequences//peers + (i < validation_sequences%peers) for i in range(peers)]
    valid = torch.cat([sample(i,n) for i,n in enumerate(counts) if n])
    return train, valid


def _local_text_data(path: Path, peers: int, sequences: int, context: int, seed: int, validation_sequences: int):
    """Byte LM over a real local text corpus; disjoint train/validation ranges.

    Contiguous per-peer train partitions retain possible distribution shifts.
    This reads at most 8MiB to avoid accidental giant dataset ingestion.
    """
    path=Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 8*1024*1024:
        raise ValueError('text file must exist and be <=8MiB')
    corpus=path.read_bytes()
    if len(corpus) < max(1024, (context+1)*peers*16):
        raise ValueError('text file is too small for disjoint peer/heldout samples')
    boundary=int(len(corpus)*.8)
    training=corpus[:boundary]
    heldout=corpus[boundary:]
    chunk=len(training)//peers
    def extract(data: bytes, count: int, init_seed: int):
        if len(data)<context+1:
            raise ValueError('text segment shorter than context')
        rng=random.Random(init_seed)
        return torch.tensor([list(data[pos:pos+context+1])
            for pos in (rng.randrange(0,len(data)-context) for _ in range(count))],dtype=torch.long)
    splits=[extract(training[i*chunk:(i+1)*chunk] if i<peers-1
                    else training[i*chunk:],sequences,seed+i*7919)
            for i in range(peers)]
    validation=extract(heldout,validation_sequences,seed+99991)
    return splits, validation, {'type':'local-utf8-bytes',
                'sha256':hashlib.sha256(corpus).hexdigest(),'bytes':len(corpus),
                'partition':'contiguous non-overlapping train chunks (80%); heldout tail (20%)'}


@lru_cache(maxsize=256)
def _shuffled_indices(count: int, epoch: int, peer: int, seed: int, cycle: int):
    # Shared deterministic sampling schedule for central and federated arms.
    order=list(range(count))
    random.Random(seed + epoch * 1000003 + peer * 1009 + cycle * 7919).shuffle(order)
    return tuple(order)


def _batch(partition, epoch, step, batch_size, seed, peer, steps_per_round=1):
    """Sample WITHOUT replacement until all sequences were seen once.

    This prevents repeated random picks within a small microbatch; a new
    permutation is used only when the training steps exhaust a peer's pool.
    Central and P2P receive exactly the same IDs for a given call.
    """
    count=len(partition)
    indices=[]
    # Sampling cursor MUST span rounds: resetting it on every epoch silently
    # repeats sequences and can corrupt apparent fine-tuning progress.
    offset=(epoch*steps_per_round+step)*batch_size
    for position in range(offset,offset+batch_size):
        cycle,idx=divmod(position,count)
        indices.append(_shuffled_indices(count,0,peer,seed,cycle)[idx])
    samples=partition[indices]
    return samples[:,:-1], samples[:,1:]


def _logits(forward_output):
    """Support both raw-tensor toy models and Transformers CausalLMOutput."""
    return forward_output if isinstance(forward_output, torch.Tensor) else forward_output.logits


def _evaluate(model, heldout, batch_size=64):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for subset in heldout.split(batch_size):
            logits=_logits(model(subset[:,:-1]))
            total += F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                         subset[:,1:].reshape(-1),reduction='sum').item()
            count += subset[:,:-1].numel()
    model.train()
    return total/count


def _train_step(model, optimizer, batch, *, grad_clip=1.0):
    x,y=batch
    optimizer.zero_grad(set_to_none=True)
    logits=_logits(model(x))
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']], grad_clip,
            error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def _create_model(initial, *, vocab, width, context, kind, rank, init_seed: int):
    # Make factor initialization identical across peers in a given round and
    # isolate it from the global torch RNG. Without this, the outcome depends
    # on how many peers were constructed before each training run.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model=TinyCausalLM(vocab,width,context)
        model.load_state_dict(initial)
        adapter=UniversalAdapter(model,'offline-causal-lm','benchmark-v1',kind=kind,rank=rank)
    return model,adapter


def _delta(adapter):
    """Materialize *effective* weight deltas, never average LoHa factors."""
    return {g.canonical:(g.overlay.local.delta() - g.overlay.anchor.delta()
                          if hasattr(g.overlay,'anchor') else g.overlay.local.delta())
                          .detach().cpu().clone()
            for g in adapter.groups}


def _sum_effective_peer_updates(aggregated: dict, changes: dict):
    """FedSum: accumulate each signed-equivalent effective weight delta ONCE.

    No division by the peer count. The caller rebases every peer to the sum
    of these deltas after all local steps for the current round.
    """
    if changes.keys() != aggregated.keys():
        raise ValueError('FedSum effective weight inventory mismatch')
    for name, delta in changes.items():
        aggregated[name].add_(delta)


def _write_csv(path: Path, rows: list[dict]):
    with path.open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _plot(path: Path, rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax=plt.subplots(figsize=(8.5,4.8))
    ax.plot([r['tokens_seen'] for r in rows], [r['central_valid_loss'] for r in rows],
            marker='o',label='Centralized LoHa')
    ax.plot([r['tokens_seen'] for r in rows], [r['p2p_valid_loss'] for r in rows],
            marker='s',label='Continual peer LoHa / dense-delta SUM (FedSum)')
    ax.set(xlabel='Total training tokens (equal budget)',ylabel='Heldout cross-entropy (nats)',
           title='Fedpara offline optimizer effectiveness (NOT WebRTC)')
    if rows and 'initial_valid_loss' in rows[0]:
        ax.axhline(rows[0]['initial_valid_loss'],linestyle='--',color='gray',label='Initial frozen-base baseline')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(path,dpi=150);plt.close(fig)
    return True


def _plot_training(path: Path, rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    xs=[r['tokens_seen'] for r in rows]
    for name,label in [('central_train_batch_loss','Central batch training loss'),
                       ('p2p_train_batch_loss','P2P batch training loss'),
                       ('central_valid_loss','Central heldout CE'),
                       ('p2p_valid_loss','P2P heldout CE')]:
        ax.plot(xs,[r[name] for r in rows],label=label)
    ax.axhline(rows[0]['initial_valid_loss'],linestyle='--',color='gray',label='Initial heldout baseline')
    ax.set(xlabel='Total training tokens',ylabel='CE in nats',title='Train minibatch vs fixed heldout')
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(path,dpi=150);plt.close(fig)
    return True


def _plot_time(path: Path, rows: list[dict]):
    """Compare equal-total-token runs using explicitly idealized P2P compute."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax=plt.subplots(figsize=(8.5,4.8))
    ax.plot([r['central_compute_s'] for r in rows],
            [r['central_valid_loss'] for r in rows],marker='o',
            label='Centralized: measured training compute')
    ax.plot([r['p2p_compute_s_div_peers'] for r in rows],
            [r['p2p_valid_loss'] for r in rows],marker='s',
            label='P2P: training compute / peers (idealized)')
    ax.set(xlabel='Training compute seconds (network/synchronization excluded)',
           ylabel='Heldout cross-entropy (nats)',
           title='Equal total tokens; ideal-parallel P2P timing, NOT wall time')
    if rows and 'initial_valid_loss' in rows[0]:
        ax.axhline(rows[0]['initial_valid_loss'],linestyle='--',color='gray',label='Initial frozen-base baseline')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(path,dpi=150);plt.close(fig)
    return True


def _sample_windows(tokens: list[int], n: int, context: int, seed: int):
    """Deterministic GPT-2 BPE-token windows; no target/label leakage across splits."""
    if len(tokens) < context + 1:
        raise ValueError('not enough GPT-2 tokens for one context window')
    rng = random.Random(seed)
    starts = [rng.randrange(len(tokens) - context) for _ in range(n)]
    return torch.tensor([tokens[i:i+context+1] for i in starts], dtype=torch.long)


def _tokenized_file_data(path, tokenizer, peers, sequences, context, seed, validation_sequences):
    """Split text BEFORE tokenizing, preventing cross-boundary training windows."""
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('text file must exist and be <=8MiB')
    raw = path.read_bytes()
    if len(raw) < 1024:
        raise ValueError('local corpus needs at least 1024 bytes')
    boundary = int(len(raw) * .8)
    # Decode each portion independently; one partial Unicode code point is replaced.
    train_raw, val_raw = raw[:boundary], raw[boundary:]
    parts = [train_raw[i * len(train_raw)//peers: (i+1)*len(train_raw)//peers].decode('utf8','replace')
             for i in range(peers)]
    train = [_sample_windows(tokenizer.encode(t, add_special_tokens=False), sequences,
                             context, seed+i*7919) for i,t in enumerate(parts)]
    valid_tokens = tokenizer.encode(val_raw.decode('utf8','replace'), add_special_tokens=False)
    valid = _sample_windows(valid_tokens, validation_sequences, context, seed+99991)
    return train, valid, {'type':'local-gpt2-bpe','sha256':hashlib.sha256(raw).hexdigest(),
                          'bytes':len(raw),'heldout_disjoint':True,
                          'partition':'80/20 split BEFORE tokenization; disjoint contiguous peer chunks'}


def _builtin_story_data(tokenizer, peers, sequences, context, seed, validation_sequences):
    """Quick offline *smoke-test* corpus, not scientific evidence of LM quality."""
    people=['Mira','Sam','Ari','Kim','Jo','Rin','Eva','Leo']
    places=['garden','village','river','forest','library','school','farm','town']
    objects=['kite','book','apple','map','boat','clock','flower','lantern']
    def make_story(j: int, *, validation: bool):
        person=people[(j*5+int(validation)*3)%len(people)]
        place=places[(j*3+int(validation)*5)%len(places)]
        thing=objects[(j*7+int(validation))%len(objects)]
        if validation:
            return f'One rainy morning {person} went to the {place}. There was a {thing} beside the path. ' \
                   f'The clouds parted and everyone smiled. {person} shared the {thing} with a friend.\n'
        return f'Once upon a time {person} visited the {place}. A small {thing} was waiting there. ' \
               f'{person} carried the {thing} home and told a gentle story about it.\n'
    # Make each peer's data non-IID by using a contiguous stretch of examples.
    train=[]
    for peer in range(peers):
        corpus=''.join(make_story(peer*180+j,validation=False) for j in range(180))
        ids=tokenizer.encode(corpus,add_special_tokens=False)
        train.append(_sample_windows(ids,sequences,context,seed+peer*7919))
    valid_ids=tokenizer.encode(''.join(make_story(j,validation=True) for j in range(180)),
                               add_special_tokens=False)
    valid=_sample_windows(valid_ids,validation_sequences,context,seed+99991)
    return train, valid, {'type':'builtin-story-smoke-test', 'heldout_disjoint':True,
                          'note':'Short synthetic prose; use --dataset tinystories for meaningful heldout evaluation.'}


def _tinystories_data(tokenizer, peers, sequences, context, seed, validation_sequences):
    """Small bounded streaming subset from true, separate HF train/validation splits."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError('TinyStories streaming requires: pip install "datasets>=4,<5"') from e

    def extract(split, wanted):
        # Hard limit prevents unbounded streaming if samples are short/invalid.
        print(f'DATASET TinyStories: fetching split={split} samples={wanted}',flush=True)
        ds=load_dataset('roneneldan/TinyStories',revision=TINYSTORIES_DATASET_REVISION,
                        split=split,streaming=True)
        result=[]
        for ix, row in enumerate(itertools.islice(ds, wanted*8)):
            story=row.get('text','')
            if not isinstance(story,str):
                continue
            ids=tokenizer.encode(story,add_special_tokens=False)
            if len(ids) < context+1:
                continue
            # Deterministic random window per example; no overlap between splits.
            rng=random.Random(seed+ix*1009+(10**9 if split=='validation' else 0))
            start=rng.randrange(len(ids)-context)
            result.append(ids[start:start+context+1])
            if len(result)%256==0:
                print(f'DATASET TinyStories: {split} {len(result)}/{wanted}',flush=True)
            if len(result)>=wanted:
                break
        if len(result)<wanted:
            raise RuntimeError(f'TinyStories {split}: only {len(result)}/{wanted} usable sequences')
        return torch.tensor(result,dtype=torch.long)
    samples=extract('train',peers*sequences)
    train=[samples[i*sequences:(i+1)*sequences].clone() for i in range(peers)]
    valid=extract('validation',validation_sequences)
    return train,valid,{'type':'hf-roneneldan-TinyStories',
                        'revision':TINYSTORIES_DATASET_REVISION, 'train_split':'train',
                        'heldout_split':'validation', 'heldout_disjoint':True,
                        'streaming':True,'bounded_sequences_per_peer':sequences,
                        'validation_sequences':len(valid), 'total_training_sequences':len(samples)}


def _create_pretrained_model(initial_model, *, kind, rank, init_seed,
                             target_scope='gpt_neo_linear', init_style='balanced',
                             model_id=TINYSTORIES_1M_ID, model_revision=TINYSTORIES_1M_REVISION):
    # Copy from a single loaded checkpoint, NEVER redownload per peer.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model=copy.deepcopy(initial_model)
        model.train()
        adapter=UniversalAdapter(model,model_id,model_revision,kind=kind,rank=rank,
                                 target_scope=target_scope,init_style=init_style)
        # Excluded pretrained weights must not accumulate grads; this also
        # saves memory when 4-16 peer copies are simulated on CPU.
        trainables={id(p) for p in adapter.trainable_parameters()}
        for param in model.parameters():
            if id(param) not in trainables:
                param.requires_grad_(False)
    return model,adapter


def scheduled_lr(base_lr: float, step: int, total_steps: int,
                 warmup_ratio: float = .05, min_lr_ratio: float = .1) -> float:
    """Linear warmup, cosine anneal. 1-indexed optimizer update count."""
    if not 1 <= step <= total_steps:
        raise ValueError('scheduler step out of range')
    warmup = min(max(1, round(total_steps * warmup_ratio)), total_steps)
    if step <= warmup:
        return base_lr * step / warmup
    progress = (step-warmup) / max(1, total_steps-warmup)
    return base_lr * (min_lr_ratio + (1-min_lr_ratio)*.5*(1+math.cos(math.pi*progress)))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def run_effectiveness_benchmark(*, peers=4, rounds=5, local_steps=2, batch_size=4,
                                seed=20261023, lr=None, kind='loha', rank=None,
                                warmup_ratio=0.05, min_lr_ratio=0.1,
                                grad_clip=1.0, target_scope=None, optimizer='adamw',
                                muon_momentum=.9, muon_weight_decay=0.,
                                width=24, context=12, vocab=32, sequences=None,
                                validation_sequences=None, output_dir: str | Path | None = None, threads=1,
                                text_file: str | Path | None = None,
                                model='toy', dataset='auto', initialization=None,
                                reinit_every=20):
    """Compare persistent-state LoHa/AdamW vs ideal synchronous SUM deltas.

    model='tinystories-1m' uses GPT-Neo (width 64, 8 layers); RANDOM frozen base by default.
    dataset='tinystories' streams bounded examples from separate train/validation
    splits. 'builtin' is a no-download smoke test (checkpoint still downloads).
    """
    if type(reinit_every) is not int or not 0 <= reinit_every <= 200:
        raise ValueError('reinit_every must be 0..200 rounds')
    if initialization is None:
        initialization = 'random'
    if initialization not in ('random','pretrained') or (model == 'toy' and initialization!='random'):
        raise ValueError('initialization must be random/pretrained (toy supports random only)')
    if sequences is None:
        sequences = 2048 if model=='tinystories-1m' else 64
    if validation_sequences is None:
        validation_sequences = 1024 if model=='tinystories-1m' else 64
    if rank is None:
        rank = 16 if model=='tinystories-1m' else 4
    if lr is None:
        lr = (.01 if model=='tinystories-1m' else .003) if optimizer=='lora-muon' else (1e-4 if model=='tinystories-1m' else 3e-3)
    if target_scope is None:
        target_scope = 'gpt_neo_full' if model=='tinystories-1m' else 'all'
    if optimizer not in ('adamw','lora-muon') or (optimizer=='lora-muon' and kind!='lora'):
        raise ValueError('LoRA-Muon is only defined with kind=lora; use AdamW for LoHa')
    if not (0 <= muon_momentum < 1 and 0 <= muon_weight_decay and lr*muon_weight_decay < 1):
        raise ValueError('invalid Muon momentum/weight decay')
    if not (0 <= warmup_ratio <= .5 and 0 < min_lr_ratio <= 1 and grad_clip > 0 and
            target_scope in ('all','gpt_neo_linear','gpt_neo_full')):
        raise ValueError('invalid optimizer policy')
    if not (2 <= peers <= 16 and 1 <= rounds <= 200 and 1 <= local_steps <= 40 and
            1 <= batch_size <= 64 and 2 <= rank <= 32 and 1 <= sequences <= 8192 and
            16 <= validation_sequences <= 8192 and
            0 < lr <= .1 and kind in ('loha','lora') and 1 <= threads <= 64 and
            isinstance(seed,int) and 0 <= seed < 2**63):
        raise ValueError('benchmark argument outside safe range')
    if model not in ('toy','tinystories-1m') or dataset not in ('auto','builtin','tinystories'):
        raise ValueError('model must be toy/tinystories-1m and dataset auto/builtin/tinystories')
    if model=='toy' and not (16 <= width <= 128 and 4 <= context <= 64 and 16 <= vocab <= 256):
        raise ValueError('toy dimensions outside safe range')
    if not 4 <= context <= (2048 if model=='tinystories-1m' else 64):
        raise ValueError('context out of range')
    if model=='toy' and (rank>width or rank>vocab):
        raise ValueError('rank must not exceed toy model dimensions')
    if model=='toy' and text_file is not None and vocab != 256:
        raise ValueError('local byte corpus requires --vocab 256')
    if model=='toy' and dataset!='auto':
        raise ValueError('the toy model uses synthetic Markov data or --text-file')
    if text_file is not None and dataset not in ('auto','builtin'):
        raise ValueError('choose either --text-file or --dataset tinystories')

    old_threads=torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        torch.manual_seed(seed)
        if model=='toy':
            initial_model=TinyCausalLM(vocab,width,context)
            base={k:v.detach().cpu().clone() for k,v in initial_model.state_dict().items()}
            if text_file is None:
                train,valid=_data(peers,sequences,context,vocab,seed,validation_sequences)
                dataset_info={'type':'synthetic-non-iid-markov','heldout_disjoint':True}
            else:
                train,valid,dataset_info=_local_text_data(
                    Path(text_file),peers,sequences,context,seed,validation_sequences)
            make_trainable=lambda: _create_model(base,vocab=vocab,width=width,
                    context=context,kind=kind,rank=rank,init_seed=seed+12345)
            make_evaluator=lambda: TinyCausalLM(vocab,width,context)
            model_id='fedpara:toy-causal-lm'
            model_revision='benchmark-v1'
        else:
            try:
                from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                raise RuntimeError('TinyStories-1M requires: pip install "transformers>=4.45"') from e
            print(f'MODEL: {TINYSTORIES_1M_ID} architecture; '
                  f'initialization={initialization}; seed={seed}; '
                  f'revision={TINYSTORIES_1M_REVISION}',flush=True)
            if initialization == 'random':
                # from_config constructs all parameter tensors afresh. NEVER
                # download or restore TinyStories pretrained model weights.
                config=AutoConfig.from_pretrained(
                    TINYSTORIES_1M_ID,revision=TINYSTORIES_1M_REVISION,
                    trust_remote_code=False)
                config._attn_implementation='eager'
                initial_model=AutoModelForCausalLM.from_config(
                    config,trust_remote_code=False)
            else:
                # Optional control experiment. This repository has .bin, so
                # keep the safe weights-only restriction when using it.
                major, minor = [int(x) for x in torch.__version__.split('+')[0].split('.')[:2]]
                if (major, minor) < (2, 6):
                    raise RuntimeError('Pretrained .bin requires torch>=2.6 safe loader')
                initial_model=AutoModelForCausalLM.from_pretrained(
                    TINYSTORIES_1M_ID,revision=TINYSTORIES_1M_REVISION,
                    trust_remote_code=False,use_safetensors=False,
                    weights_only=True,attn_implementation='eager')
            tokenizer=AutoTokenizer.from_pretrained(
                TINYSTORIES_1M_ID,revision=TINYSTORIES_1M_REVISION,trust_remote_code=False)
            config=initial_model.config
            if getattr(config,'model_type',None)!='gpt_neo':
                raise RuntimeError(f'Expected GPT-Neo TinyStories-1M, got {getattr(config,"model_type",None)!r}')
            # Deterministic comparisons: no stochastic dropout, no cache.
            for layer in initial_model.modules():
                if isinstance(layer, nn.Dropout):
                    layer.p=0.0
            config.use_cache=False
            initial_model.eval()
            if context > int(getattr(config,'max_position_embeddings',0)):
                raise ValueError('context exceeds TinyStories-1M positional embeddings')
            vocab=int(config.vocab_size)
            width=int(config.hidden_size)
            base={k:v.detach().cpu().clone() for k,v in initial_model.state_dict().items()}
            if text_file is not None:
                train,valid,dataset_info=_tokenized_file_data(
                    text_file,tokenizer,peers,sequences,context,seed,validation_sequences)
            elif dataset in ('auto','tinystories'):
                train,valid,dataset_info=_tinystories_data(
                    tokenizer,peers,sequences,context,seed,validation_sequences)
            else:
                train,valid,dataset_info=_builtin_story_data(
                    tokenizer,peers,sequences,context,seed,validation_sequences)
            make_trainable=lambda: _create_pretrained_model(initial_model,kind=kind,
                      rank=rank,init_seed=seed+12345,target_scope=target_scope,
                      init_style='balanced',
                      model_id=TINYSTORIES_1M_ID,model_revision=TINYSTORIES_1M_REVISION)
            make_evaluator=lambda: copy.deepcopy(initial_model)
            model_id=TINYSTORIES_1M_ID
            model_revision=TINYSTORIES_1M_REVISION
        if len(valid)!=validation_sequences or any(len(x)!=sequences for x in train):
            raise RuntimeError('dataset volume mismatch; refusing to benchmark incomplete data')
        dataset_info.update({'training_sequences_per_peer':sequences,
                             'training_sequences_total':sum(len(x) for x in train),
                             'validation_sequences':len(valid),
                             'sequence_length':context+1})
        print(f'DATASET READY train={sum(len(x) for x in train)} '
              f'({sequences}/peer x {peers}), validation={len(valid)}, '
              f'context={context}',flush=True)
        print(f'TRAINING BUDGET sample_draws_per_peer={rounds*local_steps*batch_size} '
              f'from {sequences} available; increase --rounds or --local-steps '
              f'to use more of the training pool',flush=True)
        from .lora_muon import make_adapter_optimizer
        central,ca=make_trainable()
        central_optim=make_adapter_optimizer(ca, optimizer=optimizer,lr=lr,weight_decay=muon_weight_decay if optimizer=='lora-muon' else 0.,muon_momentum=muon_momentum)
        from .continual import enable_anchors
        peer_trainers=[]
        for peer in range(peers):
            local,ad=make_trainable()
            enable_anchors(ad)
            opt=make_adapter_optimizer(ad,optimizer=optimizer,lr=lr,weight_decay=muon_weight_decay if optimizer=='lora-muon' else 0.,muon_momentum=muon_momentum)
            peer_trainers.append((local,ad,opt))
        # Tied weights (embedding/lm_head) appear more than once in a state_dict,
        # but ONLY one canonical adapter group is updated for each unique Parameter.
        federated_base={k:v.clone() for k,v in base.items()}
        epoch_rows=[]
        central_seconds=0.; peer_serial_seconds=0.; peer_parallel_critical_seconds=0.
        aggregation_seconds=0.; transmitted_bytes=0; factor_bytes=0; train_tokens=0
        base_loss=_evaluate(initial_model,valid,batch_size=8 if model=='tinystories-1m' else 64)
        total_central_steps=rounds*local_steps*peers
        total_local_steps=rounds*local_steps
        central_step_index=0
        peer_step_indices=[0]*peers
        for epoch in range(rounds):
            t=time.perf_counter()
            # Interleave peer partitions: processing an entire peer's non-IID
            # data consecutively creates a systematic last-peer recency bias.
            central_round_losses=[]
            for step in range(local_steps):
                for peer in range(peers):
                    central_step_index += 1
                    set_optimizer_lr(central_optim,scheduled_lr(
                        lr,central_step_index,total_central_steps,warmup_ratio,min_lr_ratio))
                    central_round_losses.append(_train_step(
                        central,central_optim,_batch(train[peer],epoch,step,batch_size,seed,peer,local_steps),
                        grad_clip=grad_clip))
            central_seconds+=time.perf_counter()-t
            aggregated={g.canonical:torch.zeros_like(federated_base[g.canonical]) for g in ca.groups}
            times=[]
            peer_round_losses=[]
            for peer,(local,ad,optim) in enumerate(peer_trainers):
                t=time.perf_counter()
                for step in range(local_steps):
                    peer_step_indices[peer] += 1
                    set_optimizer_lr(optim,scheduled_lr(
                        lr,peer_step_indices[peer],total_local_steps,
                        warmup_ratio,min_lr_ratio))
                    peer_round_losses.append(_train_step(
                        local,optim,_batch(train[peer],epoch,step,batch_size,seed,peer,local_steps),
                        grad_clip=grad_clip))
                times.append(time.perf_counter()-t)
                changes=_delta(ad)
                factor_bytes+=2*len(ad.export(f'peer{peer}',seq=epoch+1,dtype='float32'))
                _sum_effective_peer_updates(aggregated, changes)
                transmitted_bytes+=sum(x.numel()*x.element_size() for x in changes.values())
            peer_serial_seconds+=sum(times)
            peer_parallel_critical_seconds+=max(times)
            t=time.perf_counter()
            with torch.no_grad():
                # The entire model is on CPU. Rebase only canonical *trainable*
                # weights; keep frozen buffers and any tied aliases unchanged.
                for name,update in aggregated.items():
                    federated_base[name].add_(update)
                for _,ad,_ in peer_trainers:
                    for g in ad.groups:
                        module,_,attr=g.aliases[0].rpartition('.')
                        base_param=ad.model.get_submodule(module).parametrizations[attr].original
                        base_param.copy_(federated_base[g.canonical].to(base_param.device))
                        for j,part in enumerate(g.overlay.local.parts):
                            getattr(g.overlay.anchor,f'part{j}').copy_(part.detach())
            aggregation_seconds+=time.perf_counter()-t
            refreshed = bool(reinit_every and (epoch+1) % reinit_every == 0)
            if refreshed:
                from .relora import merge_and_reset_central, reset_committed_factors
                from .relora import reinit_seed
                from .lora_muon import make_adapter_optimizer
                # Merge central uncommitted LoHa effect into its frozen base;
                # reset factor coordinates and optimizer moments WITHOUT
                # altering any already-learned effective weights.
                merge_and_reset_central(ca, seed=reinit_seed('benchmark-central',str(seed),epoch+1))
                central_optim=make_adapter_optimizer(ca,optimizer=optimizer,lr=lr,
                    weight_decay=muon_weight_decay if optimizer=='lora-muon' else 0.,
                    muon_momentum=muon_momentum)
                new_peers=[]
                for peer,(local,ad,_opt) in enumerate(peer_trainers):
                    reset_committed_factors(ad,seed=reinit_seed('benchmark-peer',f'{seed}-{peer}',epoch+1))
                    fresh_opt=make_adapter_optimizer(ad,optimizer=optimizer,lr=lr,
                        weight_decay=muon_weight_decay if optimizer=='lora-muon' else 0.,
                        muon_momentum=muon_momentum)
                    new_peers.append((local,ad,fresh_opt))
                peer_trainers=new_peers
            evaluated=make_evaluator()
            # Restore canonical learned base weights without accidentally losing
            # tied HF parameters; aliases point at the same original Parameter.
            with torch.no_grad():
                for name, update in aggregated.items():
                    module,_,attr=name.rpartition('.')
                    getattr(evaluated.get_submodule(module),attr).copy_(federated_base[name])
            train_tokens+=peers*local_steps*batch_size*context
            row={'round':epoch+1,'tokens_seen':train_tokens,
                 'relora_reinitialized':refreshed,
                 'initial_valid_loss':base_loss,
                 'central_train_batch_loss':sum(central_round_losses)/len(central_round_losses),
                 'p2p_train_batch_loss':sum(peer_round_losses)/len(peer_round_losses),
                 'central_lr':central_optim.param_groups[0]['lr'],
                 'p2p_lr':peer_trainers[0][2].param_groups[0]['lr'],
                 'central_valid_loss':_evaluate(central,valid,batch_size=8 if model=='tinystories-1m' else 64),
                 'p2p_valid_loss':_evaluate(evaluated,valid,batch_size=8 if model=='tinystories-1m' else 64),
                 'central_compute_s':central_seconds,
                 'p2p_total_compute_s':peer_serial_seconds,
                 'p2p_compute_s_div_peers':peer_serial_seconds/peers,
                 'p2p_max_peer_s_sum':peer_parallel_critical_seconds,
                 'p2p_aggregation_s':aggregation_seconds}
            epoch_rows.append(row)
            print(f'ROUND {epoch+1}/{rounds}  tokens={train_tokens}  '
                  f'central_CE={row["central_valid_loss"]:.5f}  '
                  f'P2P_CE={row["p2p_valid_loss"]:.5f}',flush=True)
        last=epoch_rows[-1]
        out={'status':'OFFLINE_LOHA_EFFECTIVENESS' if optimizer=='adamw' and kind=='loha' else 'OFFLINE_LOW_RANK_EFFECTIVENESS',
             'scope':'in-process synchronous ideal-peer simulation; NOT live P2P/WebRTC',
             'configuration':{'peers':peers,'rounds':rounds,'local_steps':local_steps,
                 'batch_size':batch_size,'seed':seed,'lr':lr,'kind':kind,'rank':rank,
                 'width':width,'context':context,'vocab':vocab,'sequences_per_peer':sequences,
                 'validation_sequences':validation_sequences,'training_sequences_total':sequences*peers,
                 'sample_draws_per_peer':rounds*local_steps*batch_size,
                 'threads':threads, 'warmup_ratio':warmup_ratio,'min_lr_ratio':min_lr_ratio,
                 'optimizer':optimizer, 'muon_momentum':muon_momentum,
                 'muon_weight_decay':muon_weight_decay,
                 'grad_clip':grad_clip,'target_scope':target_scope,
                 'init_style':'balanced' if model=='tinystories-1m' else 'legacy',
                 'initialization':initialization,
                 'p2p_aggregation':'sum', 'p2p_aggregation_scale':1.0,
                 'reinit_every':reinit_every, 'reinit_unit':'completed rounds',
                 'base_weights_frozen':True,
                 'model':model,'model_id':model_id,'model_revision':model_revision,
                 'dataset':dataset_info,'heldout_disjoint':True,
                 'adapter_counts':ca.counts()},
             'fairness':{'identical_genesis':True,'identical_total_tokens':True,
                 'identical_training_samples':True,
                 'identical_initial_adapter_factors_at_round_1':True,
                 'peer_factors_seeded_identically_at_initialization':True,
                 'central_optimizer':f'one continuing {optimizer} adapter',
                 'peer_optimizer':f'persistent {optimizer} moments and {kind} factors within each refresh cycle; reset every {reinit_every} rounds (0=off)',
                 'central_optimizer_refresh':'central LoHa factors merged to frozen base, then factors and optimizer restarted at same cadence',
                 'training_sample_order':'central round-robin across peers; P2P local order by peer',
                 'peer_aggregation':'FedSum: SUM of effective deltas, no division by peers',
                 'sampling':'no replacement across rounds until each peer dataset exhausted',
                 'scheduler':'per-arm fractional-progress linear warmup and cosine decay',
                 'compute_time_rule':'sum(sequentially measured peer training seconds) / peers',
                 'caveat':'dividing by peers ASSUMES perfectly parallel equal-speed devices; not measured wall time. Max-per-round is an alternative lower-bound estimate; communication excluded.'},
             'loss':{'initial':base_loss,'central':last['central_valid_loss'],
                     'p2p':last['p2p_valid_loss'],
                     'initial_perplexity':math.exp(base_loss),
                     'central_perplexity':math.exp(last['central_valid_loss']),
                     'p2p_perplexity':math.exp(last['p2p_valid_loss'])},
             'compute':{'central_s':central_seconds,'p2p_serial_s':peer_serial_seconds,
                  'p2p_divided_by_peers_s':peer_serial_seconds/peers,
                  'p2p_max_peer_critical_path_s':peer_parallel_critical_seconds,
                  'p2p_aggregation_s':aggregation_seconds,
                  'network_s':'not measured',
                  'approx_dense_update_bytes_no_network':transmitted_bytes,
                  'estimated_before_after_factor_patch_bytes':factor_bytes},
             'trace':epoch_rows,
             'limitations':['No actual network or churn; not a Fedpara signed-event Gossip benchmark.',
                            'FedSum is an intentionally stronger update than FedAvg: effective update magnitude can grow with the number of peers; divergence is possible.',
                            'Random TinyStories-1M has a FROZEN random base: only LoHa/LoRA learns, with continual before/after factor deltas merged into the base each synchronization round.' if model=='tinystories-1m' and initialization=='random' else ('Pretrained weights: fine-tuning experiment.' if model=='tinystories-1m' else 'Toy Markov task, not text generation.'),
                            'Live asynchronous gossip does not use this offline benchmark synchronization barrier.',
                            'Live reset is per signed LOCAL publish count; offline reset is per synchronous round. Not identical time intervals.',
                            'Compute / peers is idealized; data loading, evaluation, model cloning and communication overhead excluded.']}
        if output_dir is not None:
            target=Path(output_dir).expanduser().resolve()
            target.mkdir(parents=True,exist_ok=True)
            _write_csv(target/'effectiveness-trace.csv',epoch_rows)
            out['chart_created']=_plot(target/'effectiveness-loss.png',epoch_rows)
            out['time_chart_created']=_plot_time(target/'effectiveness-time.png',epoch_rows)
            out['training_chart_created']=_plot_training(target/'effectiveness-training.png',epoch_rows)
            out['artifacts']={name:str(target/name) for name in
                 ['effectiveness-result.json','effectiveness-trace.csv']+
                 (['effectiveness-loss.png'] if out['chart_created'] else [])+
                 (['effectiveness-time.png'] if out['time_chart_created'] else [])+
                 (['effectiveness-training.png'] if out['training_chart_created'] else [])}
            with (target/'effectiveness-result.json.tmp').open('w',encoding='utf8') as fh:
                json.dump(out,fh,ensure_ascii=False,indent=2)
            os.replace(target/'effectiveness-result.json.tmp',target/'effectiveness-result.json')
        return out
    finally:
        torch.set_num_threads(old_threads)
