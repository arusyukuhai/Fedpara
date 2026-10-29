"""CLI entry point: python -m patchlib.cli {inspect,train,train-p2p,exchange,merge}"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
import os
import sys
import hashlib
import importlib.util
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from .core import UniversalAdapter, PatchError, decode_patch
from .p2p import PatchNode, exchange_file, exchange_state
from .state import export_state, apply_state, verify_state, inspect_state, semantic_state_sha256
from .identity import generate_keypair, signer_from_pem, verifier_from_pem
from .shards import (shard_state, export_state_sharded, read_manifest, verify_shards,
                     assemble_state, DEFAULT_SHARD_BYTES, shard_map,
                     apply_sharded_state, semantic_shard_sha256)
from .shardp2p import ShardNode, exchange_shards

AUTO_CLASSES = {
    'base': 'AutoModel',
    'causal-lm': 'AutoModelForCausalLM',
    'masked-lm': 'AutoModelForMaskedLM',
    'seq2seq-lm': 'AutoModelForSeq2SeqLM',
    'sequence-classification': 'AutoModelForSequenceClassification',
    'token-classification': 'AutoModelForTokenClassification',
    'question-answering': 'AutoModelForQuestionAnswering',
    'image-classification': 'AutoModelForImageClassification',
    'audio-classification': 'AutoModelForAudioClassification',
    'speech-seq2seq': 'AutoModelForSpeechSeq2Seq',
    'object-detection': 'AutoModelForObjectDetection',
    'semantic-segmentation': 'AutoModelForSemanticSegmentation',
}


def _untrusted_guard(args):
    """A policy check, NOT a security boundary: sandbox isolation is external."""
    custom = bool(getattr(args,'trust_remote_code',False) or getattr(args,'factory_file',None))
    if not custom:
        return
    if os.environ.get('FEDPARA_ISOLATED_RUNTIME') != '1':
        raise PermissionError('Custom Python code requires the offline Docker sandbox; see sandbox/run_untrusted.sh')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        raise PermissionError('Custom Python code can only run in explicitly offline runtime')
    if getattr(args,'allow_legacy_bin',False):
        raise PermissionError('Pickle/legacy checkpoint loading forbidden for untrusted code')
    if getattr(args,'trust_remote_code',False):
        if not re.fullmatch(r'[a-fA-F0-9]{40}', args.revision):
            raise PermissionError('Custom Hugging Face remote code requires a pinned 40-hex commit')
        if not Path(args.model).is_dir():
            raise PermissionError('Custom HF code requires a predownloaded local model directory')
    if getattr(args,'cmd','') in ('train-p2p', 'train-live', 'train-elastic'):
        raise PermissionError('Cannot run untrusted model code inside the networked P2P process. Train offline and exchange files outside sandbox.')


def _create_factory(args):
    factory = Path(args.factory_file)
    if not factory.is_file() or factory.suffix != '.py':
        raise ValueError('factory-file must be a local .py file')
    spec = importlib.util.spec_from_file_location('_fedpara_user_factory', factory)
    if spec is None or spec.loader is None:
        raise ValueError('could not load factory')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # ONLY inside isolated offline container
    builder = getattr(module,'build_model',None)
    if not callable(builder):
        raise ValueError('factory-file must define build_model() -> torch.nn.Module')
    model = builder()
    if not isinstance(model, torch.nn.Module):
        raise TypeError('factory must return torch.nn.Module')
    return model

def resolve_device(device):
    """Resolve runtime device consistently; fail early for unavailable accelerators."""
    if str(device) == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    result = torch.device(device)
    if result.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS selected, but PyTorch MPS is unavailable')
    if result.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected, but no available CUDA device was found')
    return result


def input_device(model, fallback):
    """Use the compute device, not a frozen CPU-mapped disk weight."""
    for mod in model.modules():
        if getattr(mod,'_fedpara_disk_path',None):
            overlay=getattr(mod,'_fedpara_disk_overlay',None)
            if overlay is not None:
                return overlay.local.parts[0].device
            return resolve_device(fallback)
    # An arbitrary PyTorch LM may expose the HF method without implementing it.
    try:
        emb = model.get_input_embeddings() if hasattr(model,'get_input_embeddings') else None
    except (AttributeError, NotImplementedError):
        emb = None
    if emb is not None:
        for param in emb.parameters():
            if param.device.type != 'meta':
                return param.device
    for param in model.parameters():
        return param.device
    return resolve_device(fallback)


def check_adapter_placement(adapter):
    """Detect mixed-device factors *before* the first forward/backward."""
    for group in adapter.groups:
        modpath, _, name = group.aliases[0].rpartition('.')
        original = adapter.model.get_submodule(modpath).parametrizations[name].original
        module=adapter.model.get_submodule(modpath)
        expected=(resolve_device(getattr(adapter.model,'_fedpara_compute_device','cpu'))
                  if hasattr(module,'_fedpara_disk_source') else original.device)
        for factor in group.overlay.local.parts:
            if factor.device != expected:
                raise RuntimeError(f'LoRA/LoHA factor for {group.canonical} on {factor.device}; expected {expected}')


def load_model(args, *, use_adapter: bool):
    _untrusted_guard(args)
    initialization = getattr(args, 'initialization', 'pretrained')
    if initialization not in ('pretrained', 'random', 'config'):
        raise ValueError('initialization must be pretrained/random/config')
    streaming=bool(getattr(args,'streaming_base',False))
    lazy_write=bool(getattr(args,'streaming_lazy_write',False))
    page_cow=bool(getattr(args,'streaming_page_cow',False))
    factor_ledger=bool(getattr(args,'streaming_factor_ledger',False))
    if factor_ledger and (not lazy_write or page_cow):
        raise ValueError('--streaming-factor-ledger requires --streaming-lazy-write and excludes --streaming-page-cow')
    if page_cow and not lazy_write:
        raise ValueError('--streaming-page-cow requires --streaming-lazy-write')
    if page_cow and getattr(args,'generation_v2_trainer',False):
        raise ValueError('paged COW exporter is not yet certified for autonomous v2 generations')
    if lazy_write and (not streaming or getattr(args,'cmd',None)!='train-elastic'
                       or getattr(args,'mode','elastic')!='elastic'):
        raise ValueError('--streaming-lazy-write requires --streaming-base and elastic train-elastic mode')
    if lazy_write and getattr(args,'disk_linear_dir',None):
        raise ValueError('--streaming-lazy-write cannot share --disk-linear-dir scratch')
    if streaming and initialization != 'pretrained':
        raise ValueError('--streaming-base requires pretrained safetensors, not random initialization')
    if streaming and getattr(args,'factory_file',None):
        raise ValueError('--streaming-base requires a Hugging Face meta-compatible architecture')
    if streaming and (getattr(args,'state_base',None) or getattr(args,'sharded_base_manifest',None)):
        raise ValueError('--streaming-base does not yet accept a separate .fstate/sharded initial override')
    if initialization == 'config':
        from .model_design import build_random_model
        from .room_manifest import load_manifest
        manifest_path=getattr(args,'room_manifest',None)
        if not manifest_path:
            raise ValueError('config creation needs an immutable room manifest')
        manifest=load_manifest(manifest_path)
        if manifest['model']['id']!=args.model or manifest['model']['revision']!=args.revision:
            raise ValueError('config manifest/worker identity mismatch')
        mdl, revision=build_random_model(manifest,args.hf_task)
    elif initialization == 'random':
        # Create the SAME GPT-Neo architecture, but do not fetch pretrained
        # weights. This seed only governs model genesis, not data sampling.
        from .live_initialization import create_random_tinystories_model
        if getattr(args, 'factory_file', None):
            raise ValueError('random TinyStories genesis cannot use custom model factory')
        mdl, revision = create_random_tinystories_model(
            args.model, args.revision, getattr(args, 'initial_seed', 42))
    elif getattr(args,'factory_file',None):
        mdl = _create_factory(args)
        factory_sha = hashlib.sha256(Path(args.factory_file).read_bytes()).hexdigest()
        revision = 'factory-sha256:'+factory_sha
    else:
        import transformers
        clsname = args.auto_class or AUTO_CLASSES[args.hf_task]
        if not (clsname == 'AutoModel' or (clsname.startswith('AutoModelFor') and clsname[12:].isalnum())):
            raise ValueError('unknown Transformers AutoModel class')
        cls = getattr(transformers,clsname)
        custom = bool(getattr(args,'trust_remote_code',False))
        if streaming:
            from .hf_config_v83 import load_hf_config
            from .streaming_base import materialize_meta_from_safetensors
            # The HF model factory must support meta construction without
            # eagerly materializing its own base weights.
            cfg=load_hf_config(args.model,revision=args.revision,
                       trust_remote_code=custom,local_files_only=custom)
            from .hf_compat_v63 import validate_training_config
            validate_training_config(cfg, args.model, streaming=True)
            with torch.device('meta'):
                mdl=cls.from_config(cfg,trust_remote_code=custom)
            if Path(args.model).is_dir():
                snapshot_dir=args.model
            else:
                from huggingface_hub import snapshot_download
                snapshot_dir=snapshot_download(args.model,revision=args.revision,
                    allow_patterns=['*.safetensors','*.json'],local_files_only=custom)
            writable=(getattr(args,'cmd','') in ('train-live','train-elastic','train-p2p') and not lazy_write)
            materialize_meta_from_safetensors(mdl,snapshot_dir,writable=writable,
                preserve_checkpoint_dtype=True,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                chunk_mib=getattr(args,'stream_chunk_mib',8),strict=True,
                hash_genesis=getattr(args,'cmd','') in ('train-live','train-elastic'))
            print('STREAMING BASE:',json.dumps(mdl._fedpara_streaming_base_info),flush=True)
        else:
            from .hf_compat_v63 import validate_training_config
            # Give actionable errors for unsupported DeepSeek FP8/MoE families.
            auto_config=getattr(transformers,'AutoConfig',None)
            config = None
            if auto_config is not None:
                from .hf_config_v83 import load_hf_config
                config = load_hf_config(args.model,revision=args.revision,
                    trust_remote_code=custom,local_files_only=custom)
                validate_training_config(config, args.model, streaming=False)
            # v4 calls this torch_dtype; v5 calls it dtype. Avoid loading a
            # BF16 Llama checkpoint as FP32 (often twice the necessary RAM).
            version=str(getattr(transformers,'__version__',''))
            dtype_arg=({'dtype':'auto'} if version.startswith('5.') else
                       {'torch_dtype':'auto'} if version.startswith('4.') else {})
            mdl = cls.from_pretrained(args.model, revision=args.revision,
                        trust_remote_code=custom,
                        local_files_only=custom,
                        use_safetensors=not args.allow_legacy_bin,
                        **({'config':config} if config is not None else {}), **dtype_arg)
        revision = getattr(mdl.config,'_commit_hash',None) or args.revision
    device = resolve_device(args.device)
    if not streaming:
        mdl.to(device)
    if getattr(args, 'auto_rank', False) or getattr(args, 'resource_report', None):
        from .device_planner import suggest_rank_and_experts
        from .moe_planner import inspect_expert_shapes
        groups = inspect_expert_shapes(mdl)
        costs = [n for group in groups for n in group.bytes_per_expert]
        recommendation = suggest_rank_and_experts(mdl, device=str(device),
            disk_dir=getattr(args,'disk_linear_dir',None) or '.',kind=args.kind,
            expert_bytes=costs if costs else None)
        from dataclasses import asdict
        from .device_planner import automatic_expert_counts
        recommendation['expert_shapes'] = [asdict(g) for g in groups]
        if groups:
            try:
                recommendation['expert_defaults_by_group'] = automatic_expert_counts(
                    groups, max(0,recommendation['budget']['usable_bytes']//2))
            except MemoryError:
                recommendation['expert_defaults_by_group'] = None
                recommendation['expert_default_warning'] = 'Cannot fit one expert per group in budget'
        if getattr(args,'resource_report',None):
            Path(args.resource_report).write_text(json.dumps(recommendation,indent=2),encoding='utf-8')
        if getattr(args,'auto_rank',False):
            if getattr(args,'cmd','') != 'train':
                raise ValueError('--auto-rank is local-only: P2P peers must agree on the exact signed rank/layout')
            if recommendation['rank_default'] is None:
                raise MemoryError('no safe automatic rank; inspect the resource report')
            args.rank = recommendation['rank_default']
        print('RESOURCE PLAN',json.dumps({'rank_default':recommendation['rank_default'],
              'expert_count_default':recommendation['expert_count_default'],
              'source':recommendation['budget']['source']}),flush=True)
    if hasattr(mdl,'config') and hasattr(mdl.config,'use_cache'):
        mdl.config.use_cache = False
    if getattr(args,'gradient_checkpointing',False):
        from .training_compat_v84 import enable_adapter_safe_checkpointing
        enable_adapter_safe_checkpointing(mdl)
    model_id = args.model_id or args.model
    if getattr(args,'state_base',None) and getattr(args,'sharded_base_manifest',None):
        raise ValueError('--state-base and --sharded-base-manifest are mutually exclusive')
    if bool(getattr(args,'sharded_base_manifest',None)) != bool(getattr(args,'sharded_base_store',None)):
        raise ValueError('--sharded-base-manifest and --sharded-base-store must both be supplied')
    if getattr(args,'sharded_base_manifest',None):
        # A generation rollover manifest can be pinned to the original room
        # revision string, while Transformers resolves a branch name to a
        # concrete commit. The model data is independently SHA256 verified.
        from .shards import read_manifest
        base_info=read_manifest(Path(args.sharded_base_manifest),
                                max_state_bytes=args.max_state_gb*1024**3)
        if base_info['revision'] != revision:
            from .room_manifest import load_manifest
            pinned=load_manifest(Path(args.room_manifest)) if getattr(args,'room_manifest',None) else None
            if (not pinned or base_info['revision']!=args.revision or
                pinned['model']['revision']!=args.revision or
                pinned['model']['id']!=model_id):
                raise ValueError('sharded base revision differs from pinned model metadata')
            revision=base_info['revision']
        snapshot=apply_sharded_state(mdl,Path(args.sharded_base_manifest),
            Path(args.sharded_base_store),model_id=model_id,revision=revision,
            max_state_bytes=args.max_state_gb*1024**3)
        revision = 'fstate-sha256:'+semantic_shard_sha256(snapshot)
    if getattr(args,'state_base',None):
        state = Path(args.state_base)
        apply_state(mdl, state, model_id=model_id,revision=revision,
                    max_bytes=args.max_state_gb*1024**3)
        # The archive ZIP header contains timestamps and peer_id can vary.
        # Bind to tensor CONTENT, not incidental archive bytes.
        revision = 'fstate-sha256:'+semantic_state_sha256(state)
    if getattr(args,'moe_plan',None):
        from .moe_light import slice_moe_modulelist
        plan=json.loads(Path(args.moe_plan).read_text(encoding='utf-8'))
        if not isinstance(plan,list) or not 1 <= len(plan) <= 1024:
            raise ValueError('MoE plan must be a nonempty list of layer plans')
        for entry in plan:
            if not isinstance(entry,dict) or set(entry)!={'experts','gate','count','seed'}:
                raise ValueError('MoE entries require experts, gate, count, seed')
            sl=slice_moe_modulelist(mdl,entry['experts'],entry['gate'],
                                    count=entry['count'],seed=entry['seed'])
            print('MoE retained expert indices:',sl.path,sl.keep,flush=True)
    if not use_adapter:
        return mdl,revision
    resolved_scope=getattr(args,'target_scope','all')
    # Automatic target detection is architecture-neutral, not GPT-Neo only.
    # The adapter records each selected layer's kind/rank in the layout digest
    # so incompatible P2P updates are rejected before application.
    args.target_scope=resolved_scope
    adapter = UniversalAdapter(mdl, model_id, revision,
                               kind=args.kind,rank=args.rank,alpha=args.alpha,
                               max_peers=args.max_peers,target_scope=resolved_scope,
                               init_style=getattr(args,'init_style','balanced'),
                               wide_lora_rank=getattr(args,'wide_lora_rank',4))
    print(f'adapter policy scope={resolved_scope} init={getattr(args,"init_style","balanced")}',flush=True)
    check_adapter_placement(adapter)
    if getattr(args,'disk_linear_dir',None) or streaming:
        from .layer_offload import install_disk_linear_offload
        dest=(getattr(args,'disk_linear_dir',None) or
              (Path(getattr(args,'streaming_scratch_dir',None) or '.') / '.fedpara-linear-pages'))
        info=install_disk_linear_offload(mdl,adapter,dest,
            min_weight_mib=(getattr(args,'disk_linear_min_mib',0.) or (1. if streaming else 0.)),
            cache_fraction=getattr(args,'disk_cache_fraction',.6),
            cache_mib=getattr(args,'disk_cache_mib',0),
            tile_mib=getattr(args,'disk_tile_mib',16),include_embeddings=streaming,
            max_dir_bytes=(getattr(args,'live_disk_linear_max_gb',32)*1024**3
                           if getattr(args,'cmd','') in ('train-live','train-elastic')
                           and getattr(args,'disk_linear_dir',None) else None))
        print('DISK LINEAR OFFLOAD:',json.dumps(info),flush=True)
        if streaming and device.type!='cpu':
            from .layer_offload import place_streaming_compute_device
            offload_cache=next((getattr(module,'_fedpara_disk_cache')
                          for module in mdl.modules() if hasattr(module,'_fedpara_disk_cache')),None)
            placement=place_streaming_compute_device(mdl,adapter,device,cache=offload_cache)
            object.__setattr__(mdl,'_fedpara_compute_device',str(device))
            print('STREAMING ACCELERATOR PLACEMENT:',json.dumps(placement),flush=True)
            # Recheck factor placements after custom CPU/GPU dispatch.
            check_adapter_placement(adapter)
    if lazy_write:
        if factor_ledger:
            from .factor_ledger_v62 import attach_factor_ledger
            manager=attach_factor_ledger(mdl,adapter,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                max_entries=getattr(args,'stream_factor_max_events',32),
                limit_gib=getattr(args,'stream_lazy_limit_gb',0))
        elif page_cow:
            from .paged_mutable_v61 import attach_paged_mutable
            manager=attach_paged_mutable(mdl,adapter,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                page_kib=getattr(args,'stream_page_kib',64),
                limit_gib=getattr(args,'stream_lazy_limit_gb',0))
        else:
            from .lazy_mutable_weights_v60 import attach_lazy_mutable
            manager=attach_lazy_mutable(mdl,adapter,
                scratch_dir=getattr(args,'streaming_scratch_dir',None),
                copy_mib=getattr(args,'stream_chunk_mib',8),
                limit_gib=getattr(args,'stream_lazy_limit_gb',0))
        print('LAZY STREAM MUTABLE:',json.dumps(manager.stats()),flush=True)
    expert_path=getattr(args,'streaming_expert_path',None)
    if expert_path:
        if not streaming:
            raise ValueError('--streaming-expert-path requires --streaming-base')
        from .expert_disk import install_streamed_experts
        count=install_streamed_experts(mdl,expert_path)
        print('STREAMED FROZEN EXPERTS:',count,expert_path,flush=True)
    print('adapter:',json.dumps({**adapter.counts(),'layout':adapter.layout,
                                   'resolved_revision':revision},indent=2))
    return mdl,adapter


def load(args):
    # In live FedSum the frozen genesis base becomes a mutable effective base.
    # Every prepared generation therefore needs its OWN mmap files. Sharing
    # --disk-linear-dir across active and shadow workers would silently corrupt
    # the active worker when the shadow truncates its file-backed weights.
    if (getattr(args,'cmd','') in ('train-live','train-elastic') and
            getattr(args,'disk_linear_dir',None)):
        import copy
        from .live_linear_workspace_v59 import allocate, attach, _release
        scratch,lease=allocate(args.disk_linear_dir)
        isolated=copy.copy(args)
        isolated.disk_linear_dir=scratch
        try:
            model,adapter=load_model(isolated,use_adapter=True)
            attach(model,scratch,lease)
            return model,adapter
        except BaseException:
            _release(lease,scratch)
            raise
    return load_model(args,use_adapter=True)


def causal_labels_from_attention(input_ids, attention_mask):
    """Preserve *real* EOS tokens when a causal LM uses EOS as padding.

    DataCollatorForLanguageModeling compares token ID with pad_token_id;
    for Llama/Qwen models whose PAD is EOS this silently drops every
    genuine EOS supervision target. Attention masks distinguish them.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError('input IDs and attention masks differ in shape')
    labels=input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    return labels


def training_batches(args):
    # A specialized SFT/PPO peer is permitted to omit a pretraining corpus.
    # Never fetch HF data for an objective with an explicit zero coefficient.
    if (getattr(args,'pt_weight',1.) == 0 and
            (getattr(args,'sft_jsonl',None) or getattr(args,'ppo_jsonl',None) or getattr(args,'sft_hf_repo','') or getattr(args,'ppo_hf_repo',''))):
        while True:
            yield {}  # run_step checks pt_weight=0 before invoking the model
    from .data_sampling import peer_sampling_seed, shuffled_cycle
    from .room_manifest import load_manifest
    manifest = (load_manifest(args.room_manifest) if getattr(args, 'room_manifest', None)
                and not (args.batches_dir or args.text_file or args.jsonl) else None)
    # Same room seed, but independent deterministic stream per persistent peer.
    root_seed = manifest['datasets']['seed'] if manifest is not None else 42
    data_seed = peer_sampling_seed(getattr(args, 'peer_id', None), root_seed)
    print(f'DATA SAMPLING peer={getattr(args, "peer_id", None) or "standalone"} '
          f'seed={data_seed} source={"huggingface" if manifest else "local"}', flush=True)
    if manifest is not None:
        from .hf_datasets import batches_from_room
        yield from batches_from_room(args,manifest)
        return
    if args.batches_dir:
        files = sorted(Path(args.batches_dir).glob('*.safetensors'))
        if not files:
            raise ValueError('No .safetensors batches in --batches-dir')
        for file in shuffled_cycle(files, data_seed):
            # A batch is exactly the named tensor arguments accepted by model(**batch).
            yield load_file(file)
    else:
        if args.hf_task not in ('causal-lm', 'masked-lm', 'seq2seq-lm', 'sequence-classification'):
            raise ValueError('Use --batches-dir for this task; no generic text loss is defined')
        from .hf_load_v85 import load_hf_tokenizer
        from transformers import DataCollatorForLanguageModeling
        tokenizer = load_hf_tokenizer(args.model, revision=args.revision,
                                                   trust_remote_code=bool(getattr(args,'trust_remote_code',False)),
                                                   local_files_only=bool(getattr(args,'trust_remote_code',False)))
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError('Tokenizer needs a PAD or EOS token')
            tokenizer.pad_token = tokenizer.eos_token
        if args.hf_task in ('causal-lm', 'masked-lm'):
            if not args.text_file:
                raise ValueError('--text-file required for text LM training')
            lines = [line.strip() for line in Path(args.text_file).read_text(encoding='utf8').splitlines() if line.strip()]
            if not lines:
                raise ValueError('empty text dataset')
            mlm = args.hf_task == 'masked-lm'
            collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15) if mlm else None
            local_rows=shuffled_cycle(lines, data_seed)
            for step in itertools.count():
                selected = list(itertools.islice(local_rows, args.batch_size))
                encoded = tokenizer(selected, truncation=True, max_length=args.max_length,
                                    padding='max_length', return_tensors='pt',
                                    **({'return_token_type_ids':False} if not mlm else {}))
                # Collator masks -100 on pad tokens for CLM, random masks for MLM.
                if mlm:
                    yield collator([{k:v[j] for k,v in encoded.items()} for j in range(args.batch_size)])
                else:
                    encoded['labels'] = causal_labels_from_attention(encoded['input_ids'], encoded['attention_mask'])
                    yield encoded
        else:
            if not args.jsonl:
                raise ValueError('--jsonl required for sequence/classification training')
            examples = [json.loads(line) for line in Path(args.jsonl).read_text(encoding='utf8').splitlines() if line.strip()]
            if not examples:
                raise ValueError('empty JSONL')
            local_rows=shuffled_cycle(examples, data_seed)
            for step in itertools.count():
                items = list(itertools.islice(local_rows, args.batch_size))
                enc = tokenizer([x['input'] if args.hf_task=='seq2seq-lm' else x['text'] for x in items],
                                return_tensors='pt',padding=True,truncation=True,max_length=args.max_length,
                                **({'return_token_type_ids':False} if args.hf_task=='seq2seq-lm' else {}))
                if args.hf_task == 'seq2seq-lm':
                    lab = tokenizer(text_target=[x['target'] for x in items],
                                    return_tensors='pt',padding=True,truncation=True,max_length=args.max_length)
                    labels = lab['input_ids']
                    labels[labels == tokenizer.pad_token_id] = -100
                    enc['labels'] = labels
                else:
                    enc['labels'] = torch.tensor([int(x['label']) for x in items], dtype=torch.long)
                yield enc


def run_step(model, adapter, optimizer, batch, args):
    with adapter.lock:
        model.train()
        target = input_device(model, args.device)
        batch = {k: v.to(target, non_blocking=True) if isinstance(v,torch.Tensor) else v
                 for k, v in batch.items()}
        from .model_contract_v64 import invoke_model, training_loss
        from .mixed_learning_v65 import enabled as mixed_enabled, build as build_mixer
        mixed=mixed_enabled(args)
        if mixed:
            previous=getattr(args,'_v65_mixer',None)
            old_adapter=getattr(args,'_v65_mixer_adapter',None)
            if previous is None or getattr(previous,'_closed',False) or (old_adapter is not None and old_adapter is not adapter):
                if previous is not None:
                    previous.close()  # epoch switch: discard old-policy PPO rollouts
                # A new signed generation invalidates the old Actor/Critic
                # pairing. Do not silently load a pre-v75 standalone Critic
                # checkpoint (same room id, but stale Actor baseline).
                skip_legacy=bool(getattr(args,'_v75_paired_active',False) and
                                 old_adapter is not None and old_adapter is not adapter)
                if skip_legacy:args._v75_skip_legacy_critic=True
                try:
                    args._v65_mixer=build_mixer(args)
                finally:
                    if skip_legacy:args._v75_skip_legacy_critic=False
                if getattr(args,'_v75_paired_active',False) and args._v65_mixer is not None:
                    args._v65_mixer._v75_paired=True
            args._v65_mixer_adapter=adapter
        mixer=getattr(args,'_v65_mixer',None) if mixed else None
        if mixer is not None and float(args.pt_weight)==0:
            loss=None  # intentional: no pretraining forward when its weight is zero
            outputs=None
        else:
            outputs = invoke_model(model, batch, input_map=getattr(args,'input_map',None))
            loss = training_loss(outputs,batch,task=getattr(args,'hf_task','causal-lm'),
                                 policy=getattr(args,'loss_policy','auto'))
        if mixer is not None:
            total=0.0
            if loss is not None:
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise ValueError('nonfinite PT objective; refusing mixed optimizer update')
                (loss * args.pt_weight / args.grad_accum).backward()
                total += float(loss.detach())*args.pt_weight
                mixer.counts['pt']+=1
            if mixer.sft is not None and args.sft_weight>0:
                sft_loss=mixer.sft_loss(model,target)
                if not bool(torch.isfinite(sft_loss.detach()).all()):
                    raise ValueError('nonfinite SFT objective; refusing mixed optimizer update')
                (sft_loss*args.sft_weight/args.grad_accum).backward()
                total+=float(sft_loss.detach())*args.sft_weight
            if mixer.ppo is not None and args.ppo_weight>0:
                mixer.critic_optim.zero_grad(set_to_none=True)
                ppo_actor,ppo_critic,metrics=mixer.ppo_loss(model,target)
                if not bool(torch.isfinite(ppo_actor.detach()).all() and
                            torch.isfinite(ppo_critic.detach()).all()):
                    raise ValueError('nonfinite PPO objective; refusing mixed optimizer update')
                (ppo_actor*args.ppo_weight/args.grad_accum).backward()
                (ppo_critic*args.ppo_weight).backward()
                # AdamW on nan/inf gradients can permanently corrupt the
                # locally persisted Critic even when the actor step is refused.
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in mixer.critic.parameters() if p.requires_grad):
                    mixer.critic_optim.zero_grad(set_to_none=True)
                    raise ValueError('nonfinite PPO Critic gradient; refusing optimizer update')
                if (getattr(args,'distributed_critic',False) and not getattr(args,'critic_rollout_gossip',False)
                        and getattr(args,'mode',None)=='elastic'):
                    # CPU copy under the SAME lock as the value update. SQLite
                    # commits and RTC fanout happen on the event loop, never
                    # from this model-training worker thread.
                    from .distributed_critic_v78 import critic_params
                    before=[p.detach().float().cpu().clone() for p in critic_params(mixer)]
                else:
                    before=None
                mixer.critic_optim.step()
                mixer.critic_steps+=1
                if before is not None:
                    args._v77_critic_before=before
                if (getattr(args,'critic_rollout_gossip',False) and getattr(args,'mode',None)=='elastic'
                        and mixer.reuses==1):
                    args._v80_rollout_pending=(mixer.export_rollout_v80(),mixer.cached['behavior_checkpoint_v82'])
                if mixer.critic_steps%10==0 and not getattr(mixer,'_v75_paired',False):
                    mixer.save_critic()
                total+=float(ppo_actor.detach())*args.ppo_weight
                args._v65_ppo_metrics=metrics
            # Preserve the existing GUI top-1 telemetry even in mixed mode.
            if (outputs is not None and getattr(args,'telemetry_file',None)
                    and getattr(args,'_metric_step',0)%10==0):
                with torch.no_grad():
                    from .model_contract_v64 import _loss_or_logits
                    _,logits=_loss_or_logits(outputs)
                    labels=batch.get('labels')
                    if (isinstance(logits,torch.Tensor) and logits.ndim==3 and
                            isinstance(labels,torch.Tensor) and labels.ndim==2 and
                            logits.shape[:2]==labels.shape):
                        valid=labels[:,1:]!=-100
                        if bool(valid.any().item()):
                            guess=logits[:,:-1,:].argmax(-1)
                            args._latest_accuracy=float(((guess==labels[:,1:]) & valid).sum().item()/valid.sum().item())
            args._metric_step=getattr(args,'_metric_step',0)+1
            if not math.isfinite(total):
                raise ValueError('nonfinite mixed objective')
            return total
        # Exact local minibatch token/top-1 accuracy, not validation accuracy.
        # Collected only once every ten steps; no logits are retained.
        if getattr(args, 'telemetry_file', None) and getattr(args, '_metric_step', 0) % 10 == 0:
            with torch.no_grad():
                logits = getattr(outputs, 'logits', None)
                labels = batch.get('labels')
                try:
                    if isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor):
                        if logits.ndim == 3 and labels.ndim == 2 and logits.shape[1] == labels.shape[1]:
                            predicted = logits[:, :-1].argmax(dim=-1)
                            truth = labels[:, 1:]
                        elif logits.ndim == 2 and labels.ndim == 1:
                            predicted, truth = logits.argmax(dim=-1), labels
                        else:
                            predicted = truth = None
                        if predicted is not None:
                            valid = truth != -100
                            total = int(valid.sum().item())
                            if total:
                                args._latest_accuracy = float(((predicted == truth) & valid).sum().item() / total)
                except (RuntimeError, ValueError):
                    pass
        args._metric_step = getattr(args, '_metric_step', 0) + 1
        (loss / args.grad_accum).backward()
        return float(loss.detach().cpu())


def _train_impl(args):
    model, adapter = load(args)
    base_seq = 0
    if args.resume:
        payload = Path(args.resume).read_bytes()
        base_seq = decode_patch(payload)[0]['seq']
        adapter.load_local(payload)
    if getattr(args,'disk_optimizer_dir',None):
        from .disk_adamw import DiskAdamW, adapter_named_parameters
        optimizer=DiskAdamW(adapter_named_parameters(adapter),args.disk_optimizer_dir,
                            lr=args.lr,weight_decay=args.weight_decay,
                            resume=getattr(args,'resume_disk_optimizer',False))
    else:
        optimizer=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=args.lr,
                                    weight_decay=args.weight_decay)
    iterator = training_batches(args)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps+1):
        loss = run_step(model, adapter, optimizer, next(iterator), args)
        if step % args.grad_accum == 0:
            with adapter.lock:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step % args.log_steps == 0 or step == 1:
            print(f'step {step}/{args.steps} loss={loss:.6g}', flush=True)
    # Flush incomplete gradient accumulation.
    if args.steps % args.grad_accum:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if hasattr(optimizer,'flush'): optimizer.flush()
    data = adapter.export(args.peer_id, seq=base_seq + args.steps, dtype=args.wire_dtype)
    Path(args.output).write_bytes(data)
    print('saved:', args.output, len(data), 'bytes')
    if args.full_state_output:
        export_state(model,args.full_state_output,model_id=adapter.model_id,
                     revision=adapter.revision,peer_id=args.peer_id,
                     seq=base_seq+args.steps,adapter=adapter,
                     max_bytes=args.max_state_gb*1024**3)
        print('saved effective full-state:',args.full_state_output)
    if args.shard_state_manifest or args.shard_store:
        if not args.shard_state_manifest or not args.shard_store:
            raise ValueError('both --shard-state-manifest and --shard-store are required')
        export_state_sharded(model,args.shard_store,args.shard_state_manifest,
                             model_id=adapter.model_id,revision=adapter.revision,
                             peer_id=args.peer_id,seq=base_seq+args.steps,
                             adapter=adapter,shard_bytes=args.shard_mb*1024**2,
                             max_state_bytes=args.max_state_gb*1024**3)
        print('saved sharded effective state:',args.shard_state_manifest)


async def _train_p2p_impl(args):
    model, adapter = load(args)
    base_seq = 0
    if args.resume:
        payload = Path(args.resume).read_bytes()
        base_seq = decode_patch(payload)[0]['seq']
        adapter.load_local(payload)
    if getattr(args,'disk_optimizer_dir',None):
        from .disk_adamw import DiskAdamW, adapter_named_parameters
        optimizer=DiskAdamW(adapter_named_parameters(adapter),args.disk_optimizer_dir,
                            lr=args.lr,weight_decay=args.weight_decay,
                            resume=getattr(args,'resume_disk_optimizer',False))
    else:
        optimizer=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=args.lr,
                                    weight_decay=args.weight_decay)
    iterator = training_batches(args)
    node = PatchNode(args.ws, args.room, args.token, max_bytes=args.max_patch_mb*1024*1024,
                     signer=_signer(args),verifier=_verifier(args))
    net = asyncio.create_task(node.run())
    seq = base_seq
    try:
        # Force a peer to be present before training; otherwise fast tasks can
        # finish entirely before ICE negotiation.
        wait_open = asyncio.create_task(node.connected.wait())
        done, _ = await asyncio.wait((wait_open, net), timeout=args.connect_timeout,
                                      return_when=asyncio.FIRST_COMPLETED)
        if net in done:
            await net
            raise ConnectionError('Signaling ended before a peer connected')
        if wait_open not in done:
            raise TimeoutError('No WebRTC peer connected within timeout')
        wait_open.cancel()
        adapter.model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, args.steps+1):
            while not node.incoming.empty():
                wire = node.incoming.get_nowait()
                try:
                    peer = adapter.import_remote(wire)
                    print(f'applied new snapshot from {peer} at step {step}',flush=True)
                except PatchError as e:
                    print('rejected remote snapshot:', e, file=sys.stderr)
            # One step runs in a worker thread while the event loop transmits.
            loss = await asyncio.to_thread(run_step, model, adapter, optimizer, next(iterator), args)
            if step % args.grad_accum == 0:
                with adapter.lock:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if step % args.sync_steps == 0:
                seq += 1
                node.publish(adapter.export(args.peer_id, seq, args.wire_dtype))
            if step % args.log_steps == 0 or step == 1:
                print(f'step {step}/{args.steps} loss={loss:.6g}',flush=True)
        if args.steps % args.grad_accum:
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if hasattr(optimizer,'flush'): optimizer.flush()
        seq += 1
        node.publish(adapter.export(args.peer_id, seq, args.wire_dtype))
        Path(args.output).write_bytes(adapter.export(args.peer_id, seq, args.wire_dtype))
        print('saved local:', args.output)
        if args.full_state_output:
            export_state(model,args.full_state_output,model_id=adapter.model_id,
                         revision=adapter.revision,peer_id=args.peer_id,
                         seq=seq,adapter=adapter,max_bytes=args.max_state_gb*1024**3)
            print('saved full-state:',args.full_state_output)
        if args.shard_state_manifest or args.shard_store:
            if not args.shard_state_manifest or not args.shard_store:
                raise ValueError('both --shard-state-manifest and --shard-store are required')
            export_state_sharded(model,args.shard_store,args.shard_state_manifest,
                                 model_id=adapter.model_id,revision=adapter.revision,
                                 peer_id=args.peer_id,seq=seq,adapter=adapter,
                                 shard_bytes=args.shard_mb*1024**2,
                                 max_state_bytes=args.max_state_gb*1024**3)
            print('saved sharded effective state:',args.shard_state_manifest)
        # Receive the other peer's final snapshot, or terminate when the
        # configured sync grace period expires.
        if args.sync_grace > 0:
            end = asyncio.get_running_loop().time() + args.sync_grace
            while asyncio.get_running_loop().time() < end and not net.done():
                try:
                    wire = await asyncio.wait_for(node.incoming.get(), timeout=min(0.5, max(0.01, end-asyncio.get_running_loop().time())))
                    try: adapter.import_remote(wire)
                    except PatchError as e: print('rejected:', e, file=sys.stderr)
                except asyncio.TimeoutError: pass
        if args.merged_output:
            adapter.merge_into_base()
            save_model(model,args.merged_output)
            print('saved merged model:',args.merged_output)
    finally:
        net.cancel()
        try: await net
        except asyncio.CancelledError: pass


def _close_mixed_trainer(args):
    """Persist the final Critic state and release HF row prefetch workers.

    This must run on success *and* failure: a ten-step periodic checkpoint
    cannot substitute for the final head/AdamW update on normal exit.
    """
    mixer=getattr(args,'_v65_mixer',None)
    if mixer is None:
        return
    try:
        mixer.close()
    finally:
        # Keep completed counters available to CLI telemetry/tests. run_step
        # notices _closed and builds a fresh instance on a later run.
        args._v65_mixer_adapter=None


def train(args):
    try:
        return _train_impl(args)
    finally:
        _close_mixed_trainer(args)


async def train_p2p(args):
    try:
        return await _train_p2p_impl(args)
    finally:
        # run_step runs in an awaited to_thread; it must be finished before
        # the local head, optimizer and iterable worker are closed.
        await asyncio.to_thread(_close_mixed_trainer,args)

def merge(args):
    model, adapter = load(args)
    if args.local_patch:
        adapter.load_local(Path(args.local_patch).read_bytes())
    for path in args.remote_patch:
        peer = adapter.import_remote(Path(path).read_bytes())
        print('loaded remote peer:',peer)
    adapter.merge_into_base()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_model(model,args.output_dir)
    print('merged model in:',args.output_dir)


def _signer(args):
    return signer_from_pem(Path(args.signing_key)) if getattr(args,'signing_key',None) else None


def _verifier(args):
    return verifier_from_pem(Path(args.peer_public_key)) if getattr(args,'peer_public_key',None) else None


def save_model(model, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if hasattr(model,'save_pretrained'):
        model.save_pretrained(str(folder),safe_serialization=True)
        # HF models normally write this themselves; enforce it on all Kagami
        # merged checkpoint exports, including third-party config serializers.
        config = getattr(model, 'config', None)
        if (config is not None and hasattr(config, 'save_pretrained')
                and isinstance(getattr(config, 'model_type', None), str)):
            from .hf_config_v83 import save_hf_config_with_model_type
            save_hf_config_with_model_type(config, folder)
    else:
        from safetensors.torch import save_file
        weights = {name: param.detach().cpu().clone() for name,param in model.state_dict().items()}
        save_file(weights,str(folder/'state.safetensors'))
        print('Factory model has no save_pretrained; state.safetensors only (factory code required to reload).')


def fetch_custom_cli(args):
    from huggingface_hub import snapshot_download
    if not re.fullmatch(r'[a-fA-F0-9]{40}',args.revision):
        raise ValueError('fetch-custom requires a pinned 40-character commit SHA')
    dest=Path(args.output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    result=snapshot_download(args.repo,revision=args.revision,local_dir=str(dest),
                             ignore_patterns=['*.bin','*.pt','*.pth','*.pkl','*.pickle','*.ckpt'])
    print('Downloaded inert files (NOT executed):', result)
    print('Review the .py files and run sandbox/run_untrusted.sh for custom code.')

def state_export_cli(args):
    model, revision = load_model(args,use_adapter=False)
    adapter = None
    if args.local_patch or args.remote_patch:
        adapter = UniversalAdapter(model, args.model_id or args.model,revision,
                                   kind=args.kind,rank=args.rank,alpha=args.alpha,max_peers=args.max_peers)
        if args.local_patch:
            adapter.load_local(Path(args.local_patch).read_bytes())
        for path in args.remote_patch:
            adapter.import_remote(Path(path).read_bytes())
    meta = export_state(model,args.output, model_id=args.model_id or args.model,
                        revision=revision,peer_id=args.peer_id,seq=args.seq,
                        adapter=adapter,max_bytes=args.max_state_gb*1024**3)
    print('saved complete model-state:',args.output,'tensors:',len(meta['tensors']))


def state_import_cli(args):
    model,revision = load_model(args,use_adapter=False)
    meta = apply_state(model,args.state,model_id=args.model_id or args.model,
                       revision=revision,max_bytes=args.max_state_gb*1024**3)
    save_model(model,args.output_dir)
    print('verified/applied',len(meta['tensors']),'tensors, saved:',args.output_dir)


def state_exchange_cli(args):
    node = PatchNode(args.ws,args.room,args.token,
                     max_bytes=args.max_state_gb*1024**3,
                     state_dir=args.recv_dir,signer=_signer(args),verifier=_verifier(args))
    result = asyncio.run(exchange_state(node,args.send,args.timeout))
    print('received:',result)

def shard_export_cli(args):
    """No giant .fstate intermediate: stream live weights into shard storage."""
    model, revision = load_model(args,use_adapter=False)
    adapter = None
    if args.local_patch or args.remote_patch:
        adapter = UniversalAdapter(model,args.model_id or args.model,revision,
                                   kind=args.kind,rank=args.rank,alpha=args.alpha,
                                   max_peers=args.max_peers)
        if args.local_patch:
            adapter.load_local(Path(args.local_patch).read_bytes())
        for path in args.remote_patch:
            adapter.import_remote(Path(path).read_bytes())
    meta = export_state_sharded(model,args.store,args.output,
                                model_id=args.model_id or args.model,
                                revision=revision,peer_id=args.peer_id,seq=args.seq,
                                adapter=adapter,shard_bytes=args.shard_mb*1024**2,
                                max_state_bytes=args.max_state_gb*1024**3)
    print('sharded effective model state:',args.output,
          'tensors:',len(meta['tensors']))


def shard_create_cli(args):
    meta = shard_state(args.state,args.store,args.output,
                       shard_bytes=args.shard_mb*1024**2,
                       max_state_bytes=args.max_state_gb*1024**3)
    unique = shard_map(meta)
    print(f'Created {args.output}: {len(meta["tensors"])} tensors, '
          f'{len(unique)} unique shards, {sum(unique.values())} unique stored bytes')


def shard_assemble_cli(args):
    meta = assemble_state(args.manifest,args.store,args.output,
                          max_state_bytes=args.max_state_gb*1024**3)
    print(f'Verified and reconstructed {args.output}: {len(meta["tensors"])} tensors')


def shard_inspect_cli(args):
    meta = read_manifest(args.manifest,max_state_bytes=args.max_state_gb*1024**3)
    unique = shard_map(meta)
    result = {'model_id':meta['model_id'],'revision':meta['revision'],
              'peer_id':meta['peer_id'],'seq':meta['seq'],
              'total_tensor_bytes':meta['total_tensor_bytes'],
              'tensors':len(meta['tensors']),'shards':sum(len(t['shards']) for t in meta['tensors']),
              'unique_shards':len(unique),'shard_bytes':meta['shard_bytes']}
    if args.verify:
        verify_shards(meta,args.store)
        result['stored_shards_verified'] = True
    print(json.dumps(result,ensure_ascii=False,indent=2))


def shard_exchange_cli(args):
    if args.send_only and not args.send:
        raise ValueError('--send-only requires --send MANIFEST')
    node = ShardNode(args.ws,args.room,args.token,store=args.store,
                     recv_dir=args.recv_dir,local_manifest=args.send,
                     max_state_bytes=args.max_state_gb*1024**3,
                     signer=_signer(args),verifier=_verifier(args),
                     request_remote=not args.send_only)
    result = asyncio.run(exchange_shards(node,receive=not args.send_only,timeout=args.timeout))
    print('received manifest:',result)
    print('transferred:',node.session.received_objects,'new shards /',
          node.session.received_payload_bytes,'bytes;',
          node.session.reused_objects,'shards reused')
    if result is not None and args.assemble_output:
        assemble_state(result,args.store,args.assemble_output,
                       max_state_bytes=args.max_state_gb*1024**3)
        print('assembled:',args.assemble_output)


def expert_smoke_cli(args):
    """Functional distributed expert forward/backward smoke test, no network."""
    from .expert_parallel import ExpertOwner,LocalEndpoint,RemoteExpert
    from torch import nn
    torch.manual_seed(1234)
    owner_model=nn.Linear(16,16)
    remote=RemoteExpert(LocalEndpoint(ExpertOwner({'experts.0':owner_model},'genesis')),
                        'experts.0','genesis')
    x=torch.randn(8,16,requires_grad=True)
    loss=remote(x).square().mean()
    loss.backward()
    print(json.dumps({'loss':loss.item(),'input_gradient':x.grad is not None,
          'owner_gradient':owner_model.weight.grad is not None,
          'transport':'in-process only'},indent=2))


def expert_checkpoint_inspect(args):
    from .expert_safetensors import checkpoint_files,expert_tensor_key
    from safetensors import safe_open
    stats={};total=0
    for path in checkpoint_files(args.checkpoint):
        with safe_open(str(path),framework='pt',device='cpu') as f:
            for key in f.keys():
                info=expert_tensor_key(key)
                if info is not None:
                    stats.setdefault(info[0],set()).add(info[1])
                total+=1
    result={'checkpoint_tensors':total,'unpacked_expert_groups':{k:sorted(v) for k,v in stats.items()}}
    print(json.dumps(result,indent=2))


def expert_plan_cli(args):
    from .expert_plan import plan_experts,plan_digest,save_plan
    layers=json.loads(Path(args.layers).read_text())
    if isinstance(layers,dict): layers=layers.get('candidates',layers.get('layers',[]))
    capacity=json.loads(Path(args.capacity).read_text())
    plan=plan_experts(args.room_id,layers,capacity,replicas=args.replicas,
                      generation=args.generation)
    print('expert plan',args.output,'sha256',save_plan(args.output,plan))


def expert_discover_cli(args):
    from .expert_parallel import discover_moe
    model,_=load_model(args,use_adapter=False)
    layers=discover_moe(model)
    print(json.dumps({'layers':layers},indent=2))


def code_register_cli(args):
    from .code_registry import register_code
    if args.file.stat().st_size>256*1024:raise ValueError('max code bundle 256KiB')
    print(json.dumps(register_code(args.registry,args.file.name,args.file.read_bytes()),indent=2))


def add_model_args(p):
    p.add_argument('--model', required=True, help='HF model ID or local pretrained directory')
    p.add_argument('--model-id', help='Stable canonical identifier; must match peers')
    p.add_argument('--factory-file', help='Untrusted custom Python build_model() factory; offline container required')
    p.add_argument('--trust-remote-code', action='store_true',help='Explicit opt-in, only in sandbox with local pinned HF snapshot')
    p.add_argument('--state-base',help='Apply verified full-state base before attaching any patches')
    p.add_argument('--sharded-base-manifest',type=Path,
                   help='Load existing model state directly from verified .fmanifest shards (no .fstate assembly)')
    p.add_argument('--sharded-base-store',type=Path,
                   help='Content-addressed shard store for --sharded-base-manifest')
    p.add_argument('--initialization',choices=('pretrained','random','config'),default='pretrained',
        help='random creates official TinyStories-1M config weights from a fixed seed (no pretrained weights)')
    p.add_argument('--initial-seed',type=int,default=42,
        help='Shared random model genesis seed; MUST match all room peers')
    p.add_argument('--max-state-gb',type=int,default=32,help='Full state file limit in GiB')
    p.add_argument('--hf-task', choices=tuple(AUTO_CLASSES), default='causal-lm')
    p.add_argument('--auto-class', help='Override with a packaged Transformers AutoModelFor* class')
    p.add_argument('--revision', default='main', help='Prefer pinned Hub commit SHA')
    p.add_argument('--kind', choices=('lora','loha'), default='loha')
    p.add_argument('--rank', type=int, default=16,help='SHARED room LoHa rank; do not auto-change between peers')
    p.add_argument('--auto-rank',action='store_true',
        help='Local train only: device-based rank default; prohibited for P2P genesis')
    p.add_argument('--resource-report',type=Path,
        help='Write device resource/rank/expert recommendations to JSON')
    p.add_argument('--disk-linear-dir',type=Path,
        help='Opt-in frozen nn.Linear disk-mapped weights; per-layer forward/backward I/O')
    p.add_argument('--live-disk-linear-max-gb',type=int,default=32,
        help='Per live model mutable base on-disk limit in GiB (not a RAM guarantee)')
    p.add_argument('--disk-linear-min-mib',type=float,default=0.,
        help='Only offload Linear weights >= this size in MiB')
    p.add_argument('--wide-lora-rank',type=int,default=4,
        help='Mixed adapter: LoRA rank for automatically detected embeddings, output heads and very rectangular matrices (default 4)')
    p.add_argument('--target-scope',choices=('all','gpt_neo_linear','gpt_neo_full','auto','linear','universal'),default='all',
        help='universal: all supported differentiable Parameters, including 1-D/N-D custom kernels. Explicit opt-in creates a NEW P2P adapter layout')
    p.add_argument('--init-style',choices=('legacy','balanced'),default='balanced',
        help='LoHa initialization: balanced avoids near-zero initial gradients')
    p.add_argument('--optimizer',choices=('adamw','lora-muon'),default='adamw',
        help='Continual adapter optimizer; LoRA-Muon requires --kind lora')
    p.add_argument('--muon-momentum',type=float,default=.9)
    p.add_argument('--muon-weight-decay',type=float,default=0.)
    p.add_argument('--moe-plan',type=Path,help='Shared, deterministic explicit ModuleList expert plan JSON (experimental)')
    p.add_argument('--alpha', type=float, default=None)
    p.add_argument('--max-peers', type=int, default=8)
    p.add_argument('--device', default='cpu', help='cpu / cuda / mps')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--allow-legacy-bin', action='store_true', help='Opt in to non-safetensors legacy checkpoint files')
    p.add_argument('--streaming-base', action='store_true',
        help='Create HF model on meta and mmap local safetensors; CPU streaming execution only')
    p.add_argument('--streaming-lazy-write',action='store_true',
        help='Experimental train-elastic: no up-front full mutable base copy; materialize each modified tensor once into private scratch')
    p.add_argument('--streaming-factor-ledger',action='store_true',
                   help='EXPERIMENTAL: retain verified before/after LoHA/LoRA factors as virtual base; compact exactly after bounded events')
    p.add_argument('--stream-factor-max-events',type=int,default=32,
                   help='maximum signed contributions per group before exact dense compaction (1..256)')
    p.add_argument('--streaming-page-cow',action='store_true',
        help='Experimental: retain only modified tensor pages on disk; no v2 auto epoch export')
    p.add_argument('--stream-page-kib',type=int,default=64,choices=(4,16,64,256,1024),
        help='Per-tensor sparse COW page size in KiB')
    p.add_argument('--streaming-expert-path',
        help='Expert ModuleList/Dict whose entirely frozen experts run on demand from CPU mappings')
    p.add_argument('--stream-lazy-limit-gb',type=int,default=0,
        help='Maximum lazy writable tensor scratch GiB (0=no hard cap; monitor disk usage)')
    p.add_argument('--streaming-scratch-dir',type=Path,
        help='Fast disk for temporary writable pages; live restarts reconstruct from immutable genesis')
    p.add_argument('--stream-chunk-mib',type=int,default=8,
        help='Max bytes per copied tensor slice during writable streaming initialization')
    p.add_argument('--disk-cache-fraction',type=float,default=.6,
        help='Fraction of free VRAM devoted to opportunistic LRU frozen-weight cache (max .85)')
    p.add_argument('--disk-cache-mib',type=int,default=0,
        help='Optional hard cache cap in MiB; 0 uses detected available memory')
    p.add_argument('--disk-tile-mib',type=int,default=16,
        help='Max frozen-weight transfer tile size in MiB for large projections')


def add_train_args(p):
    add_model_args(p)
    p.add_argument('--peer-id', help='Legacy peer identifier (auto-generated for --room-id)')
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument('--batches-dir', help='Task-neutral pretokenized .safetensors batches')
    src.add_argument('--text-file', help='Raw text for causal/MLM')
    src.add_argument('--jsonl', help='JSONL with input/target or text/label')
    p.add_argument('--room-manifest',type=Path,help='Room v2 Hugging Face dataset collection; no local text file required')
    p.add_argument('--loss-policy',choices=('auto','provided','causal','token','classification'),default='auto',
                   help='Model-neutral loss contract; auto uses returned scalar loss or known task logits')
    p.add_argument('--input-map',default='',
                   help='JSON mapping, e.g. {"input_ids":"tokens","attention_mask":null}; pin consistently across peers')
    # v65: one model, simultaneous gradient contributions from PT, SFT and PPO.
    p.add_argument('--mix-policy',type=Path,
                   help='Optional shared immutable JSON objective protocol: permits PT-only, SFT, or PPO specialized peers in one signed room')
    p.add_argument('--sft-jsonl',type=Path,help='Local JSONL {prompt,response}; alternatively use --sft-hf-repo and related flags')
    p.add_argument('--ppo-jsonl',type=Path,help='Local JSONL {prompt,reference}; alternatively use --ppo-hf-repo and related flags')
    p.add_argument('--sft-hf-repo',default='',help='Pinned Hub SFT repo')
    p.add_argument('--sft-hf-revision',default='',help='Pinned Hub SFT revision')
    p.add_argument('--sft-hf-config',default='',help='Pinned Hub SFT config')
    p.add_argument('--sft-hf-split',default='train',help='Pinned Hub SFT split')
    p.add_argument('--sft-hf-prompt-column',default='prompt',help='Pinned Hub SFT prompt_column')
    p.add_argument('--sft-hf-response-column',default='response',help='Pinned Hub SFT response_column')
    p.add_argument('--ppo-hf-repo',default='',help='Pinned Hub PPO repo')
    p.add_argument('--ppo-hf-revision',default='',help='Pinned Hub PPO revision')
    p.add_argument('--ppo-hf-config',default='',help='Pinned Hub PPO config')
    p.add_argument('--ppo-hf-split',default='train',help='Pinned Hub PPO split')
    p.add_argument('--ppo-hf-prompt-column',default='prompt',help='Pinned Hub PPO prompt_column')
    p.add_argument('--ppo-hf-reference-column',default='reference',help='Pinned Hub PPO reference_column')
    p.add_argument('--pt-weight',type=float,default=1.,help='Pretraining objective gradient coefficient')
    p.add_argument('--sft-weight',type=float,default=1.,help='Response-only SFT objective coefficient')
    p.add_argument('--ppo-weight',type=float,default=.2,help='Clipped PPO objective coefficient')
    p.add_argument('--ppo-clip',type=float,default=.2,help='PPO actor and value clipping epsilon')
    p.add_argument('--ppo-kl-coef',type=float,default=.02,help='Penalty relative to frozen rollout behavior policy')
    p.add_argument('--ppo-value-coef',type=float,default=.5,help='Auxiliary local value estimator coefficient')
    p.add_argument('--ppo-gamma',type=float,default=.99)
    p.add_argument('--ppo-gae-lambda',type=float,default=.95)
    p.add_argument('--ppo-max-tokens',type=int,default=8,help='Max autoregressive PPO rollout tokens per mixed step')
    p.add_argument('--ppo-reuse',type=int,default=2,help='Max clipped PPO updates per sampled trajectory')
    p.add_argument('--ppo-kv-cache',choices=('auto','off'),default='auto',
                   help='Use causal KV cache during no-gradient PPO rollout if forward explicitly supports it; never for PPO gradients')
    p.add_argument('--critic-kind',choices=('mlp','linear','hf'),default='mlp',help='PPO value critic: local MLP/Linear or pinned Hugging Face scalar value head')
    p.add_argument('--critic-hidden',type=int,default=32,help='Local value critic hidden width (8..512)')
    p.add_argument('--critic-lr',type=float,default=1e-3,help='Local PPO value critic learning rate')
    p.add_argument('--distributed-critic',action='store_true',help='Opt-in independent signed P2P Critic value-head training (new room; PPO peers only)')
    p.add_argument('--critic-rollout-gossip',action='store_true',
        help='EXPERIMENTAL v80: signed replay of peer PPO trajectories with receiver-side IS Critic updates (instead of sharing Critic parameter deltas)')
    p.add_argument('--critic-offpolicy',choices=('auto','gae','trajectory_is'),default='auto',
                   help='Critic value target: auto=unclipped trajectory IS in P2P mode, legacy GAE locally. No clipping; high variance possible')
    p.add_argument('--critic-hf-repo',default='',help='Pinned HF scalar value-model repo (organization/name)')
    p.add_argument('--critic-hf-revision',default='',help='Immutable 40-hex HF value critic revision')
    p.add_argument('--critic-hf-loader',choices=('sequence_classification','token_classification'),default='sequence_classification')
    p.add_argument('--critic-hf-device',choices=('cpu','cuda','mps'),default='cpu')
    p.add_argument('--critic-hf-max-length',type=int,default=512)
    p.add_argument('--critic-init',type=Path,help='Optional Kagami safetensors value-critic initial weights')
    p.add_argument('--critic-state-dir',type=Path,help='Private local value-critic optimizer+weights directory')
    p.add_argument('--ppo-reward',choices=('token_match','exact','hf_model'),default='token_match',
                   help='Verifiable ground-truth reward; never model-judged')
    p.add_argument('--reward-model-id',default='',help='Hugging Face scalar sequence-classification Reward Model repo')
    p.add_argument('--reward-model-revision',default='',help='Pinned 40-hex commit SHA of Reward Model')
    p.add_argument('--reward-model-device',choices=('cpu','cuda','mps'),default='cpu',help='Inference device, independent from actor')
    p.add_argument('--reward-model-idle-offload',action='store_true',
                   help='Opt-in move frozen HF Reward Model from accelerator to CPU between PPO scoring calls (saves resident VRAM, transfer overhead)')
    p.add_argument('--reward-model-loader',choices=('sequence_classification','auto_model'),default='sequence_classification')
    p.add_argument('--reward-model-format',choices=('plain','chat'),default='plain')
    p.add_argument('--reward-model-output',choices=('logits','score','end_scores','rewards'),default='logits')
    p.add_argument('--reward-model-index',type=int,default=0,help='Reward column (index 0 for scalar)')
    p.add_argument('--reward-model-scale',type=float,default=1.0)
    p.add_argument('--reward-model-max-length',type=int,default=512)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--disk-optimizer-dir',type=Path,
        help='Opt-in disk-mapped AdamW moments; elastic live uses journal-tip-bound working snapshots')
    p.add_argument('--disk-optimizer-max-gb',type=int,default=32,
        help='Per-instance disk AdamW moment budget in GiB (elastic mode)')
    p.add_argument('--resume-disk-optimizer',action='store_true',
        help='Resume external AdamW moments; requires matching factor layout')
    p.add_argument('--log-steps', type=int, default=10)
    p.add_argument('--resume', help='Previous local .fpatch with exactly matching model')
    p.add_argument('--wire-dtype', choices=['float32','float16'], default='float16')
    p.add_argument('--output', default='local.fpatch')
    p.add_argument('--full-state-output',help='Also emit full effective tensor state as .fstate')
    p.add_argument('--shard-state-manifest',type=Path,help='Also export effective weights as shards, no .fstate intermediate')
    p.add_argument('--shard-store',type=Path,help='Storage for --shard-state-manifest')
    p.add_argument('--shard-mb',type=int,default=4)


def add_identity(p):
    p.add_argument('--signing-key',help='Ed25519 private key PEM for outgoing file signatures')
    p.add_argument('--peer-public-key',help='Pinned Ed25519 public key PEM; reject unsigned incoming')


def main():
    parser = argparse.ArgumentParser(description='Model-independent LoRA/LoHA + Mummy/WebRTC P2P patches')
    sub = parser.add_subparsers(dest='cmd',required=True)
    insp = sub.add_parser('inspect',help='Print archive metadata')
    insp.add_argument('patch')
    archprobe=sub.add_parser('architecture-probe',help='Run a real forward+backward on an arbitrary model; no weight update or P2P')
    add_train_args(archprobe)
    trainprobe=sub.add_parser('training-preflight',help='Load one actual dataset batch and verify finite LoHA/LoRA gradients without any optimizer update or P2P')
    add_train_args(trainprobe)
    archprobe.add_argument('--probe-batch',type=Path,required=True,help='Local safetensors mapping containing a representative training batch')
    hfcheck=sub.add_parser('hf-diagnose',help='Inspect existing HF model config and local checkpoint compatibility')
    hfcheck.add_argument('--model',required=True)
    hfcheck.add_argument('--revision',default='main')
    hfcheck.add_argument('--local-files-only',action='store_true')
    trainp = sub.add_parser('train',help='Train local full-coverage adapter and export')
    add_train_args(trainp)
    live = sub.add_parser('train-p2p',help='Train and exchange snapshots concurrently over WebRTC')
    add_train_args(live)
    live.add_argument('--ws', default='ws://127.0.0.1:8080/ws')
    live.add_argument('--room', required=True)
    live.add_argument('--token', default=os.getenv('APP_TOKEN',''))
    live.add_argument('--sync-steps', type=int, default=10)
    live.add_argument('--connect-timeout', type=float, default=120)
    live.add_argument('--sync-grace', type=float, default=10)
    live.add_argument('--max-patch-mb', type=int, default=512)
    live.add_argument('--merged-output',help='Optional merged Hugging Face checkpoint directory')
    add_identity(live)
    livefed = sub.add_parser('train-live', aliases=['train-elastic'], help='Elastic (default) or synchronous streaming federated training over WebRTC')
    add_train_args(livefed)
    # The old 1e-3 + std=.02 LoHa startup was too aggressive and had extremely
    # weak first gradients. For new live rooms, match benchmark-safe settings.
    livefed.set_defaults(lr=1e-4,target_scope='auto',init_style='balanced')
    # Offload of live optimizer state is not wired to the durable journal.
    livefed.add_argument('--mode',choices=['elastic','synchronous'],default='elastic',help='Elastic is default; synchronous requires fixed roster')
    livefed.add_argument('--auto-state',action='store_true',help='Automatically create missing genesis .fstate from trusted packaged HF model; no custom code execution')
    livefed.add_argument('--room-public',action='store_true',help='Publish bearer-secret room ID in the public directory (opt-in)')
    livefed.add_argument('--max-neighbors',type=int,default=8,choices=(2,4,6,8),help='Symmetric ring/chord WebRTC links per peer; max 8')
    livefed.add_argument('--room-id',help='Room-only discovery; no keygen, --link or roster file needed')
    livefed.add_argument('--auto-tune',action=argparse.BooleanOptionalAction,default=True,help='Conservatively tune memory, batch, wire precision and local sync cadence')
    livefed.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    livefed.add_argument('--ws-backup',action='append',default=[],help='Independent emergency signaling URL (optional)')
    livefed.add_argument('--link',action='append',default=[],help='One per remote: OTHER_PEER=PAIRWISE_ROOM')
    livefed.add_argument('--peers',type=int,help='Synchronous mode only (not allowed for elastic)')
    livefed.add_argument('--expected-peers',type=int,default=2,help='Deprecated cosmetic CLI field; live membership is discovered automatically')
    livefed.add_argument('--token',default=os.getenv('APP_TOKEN',''))
    livefed.add_argument('--local-steps',type=int,default=4,help='Maximum local minibatches before publishing; receive no longer forces early seal')
    livefed.add_argument('--max-sync-steps',type=int,default=64,help='Maximum local steps per published event as room grows')
    livefed.add_argument('--fixed-sync',dest='adaptive_sync',action='store_false',help='Disable room-size-based publication spacing')
    livefed.set_defaults(adaptive_sync=True)
    livefed.add_argument('--rounds',type=int,default=0,help='Elastic: 0 runs continuously; synchronous requires > 0 rounds')
    livefed.add_argument('--reinit-every',type=int,default=20,
        help='Reinitialize committed LoHa/LoRA factors and optimizer every N local signed events; 0 disables (requires a NEW room)')
    livefed.add_argument('--round-timeout',type=float,default=600)
    livefed.add_argument('--connect-timeout',type=float,default=120)
    livefed.add_argument('--max-patch-mb',type=int,default=512)
    livefed.add_argument('--max-delta-rms',type=float,default=10.0,help='Maximum RMS of any peer update per tensor')
    livefed.add_argument('--checkpoint-every',type=int,default=0,help='Write full sharded base every N completed rounds (0 disables)')
    livefed.add_argument('--checkpoint-dir',type=Path,default=Path('live-checkpoints'))
    livefed.add_argument('--signing-key',help='Local Ed25519 private key')
    livefed.add_argument('--trusted-peer-key',action='append',default=[],help='One per peer: OTHER_PEER=PUBLIC_KEY_PEM')
    livefed.add_argument('--unsafe-unsigned',action='store_true',help='Local test only: disables peer signature checks')
    livefed.add_argument('--self-public-key',help='This peer Ed25519 public key; used to validate relayed history')
    livefed.add_argument('--roster-file',type=Path,help='Optional reloadable JSON peer roster for joining/leaving')
    livefed.add_argument('--journal',type=Path,default=Path('elastic-events.sqlite3'),help='Durable signed patch event journal')
    livefed.add_argument('--journal-sync',choices=['full','normal'],default='full',help='full: fsync each event (power-loss safe); normal: faster WAL without per-event fsync (process crash safe only, sudden power loss may lose recent events)')
    livefed.add_argument('--telemetry-file',type=Path,default=None,help='Local-only JSONL chart data; GUI supplies a unique file per run')
    livefed.add_argument('--swarm-pause-file',type=Path,help=argparse.SUPPRESS)
    livefed.add_argument('--eval-snapshot-request-file',type=Path,help=argparse.SUPPRESS)
    livefed.add_argument('--swarm-max-events',type=int,default=0,help=argparse.SUPPRESS)
    livefed.add_argument('--epoch-v2-probe-committee',type=Path,help='EXPERIMENTAL: fixed JSON list of signed peer IDs for no-op Epoch Paxos WebRTC probe; does NOT change v1 training')
    livefed.add_argument('--generation-v1-committee',type=Path,help='EXPERIMENTAL: pinned crash-fault committee for verified WebRTC generation certificate gossip. Does not automatically rebase model weights or delete history.')
    livefed.add_argument('--generation-v2-trainer',action='store_true',help='EXPERIMENTAL: permit only fully verified paged-v2 quorum certificates to drive trainer recovery and step-boundary carry')
    livefed.add_argument('--generation-v2-propose-every',type=int,default=0,help='EXPERIMENTAL: committee leader exports a fenced model checkpoint every N locally emitted signed events (0 disables; may pause local steps for large models)')
    livefed.add_argument('--inbound-queue',type=int,default=32,
        help='Maximum signed incoming updates awaiting asynchronous application (1..256)')
    livefed.add_argument('--durable-every',type=int,default=0,help='Unpublished optimizer-state checkpoint every N local steps (0=off, signed events remain FULL durable)')
    livefed.add_argument('--working-save-every',type=int,default=10,help='Persist the large local factor+AdamW working snapshot every N published updates (default 10; 1=old maximum durability)')
    livefed.add_argument('--max-journal-gb',type=int,default=0,help='Signed-event journal limit in GiB (0=unlimited by application, requires adequate disk; no silent pruning)')
    livefed.add_argument('--peer-timeout',type=float,default=45.,help='Inactive peer expires from denominator')
    livefed.add_argument('--heartbeat-interval',type=float,default=4.,help='Signed heartbeat cadence in seconds')
    livefed.add_argument('--hf-shuffle-buffer',type=int,default=512,
                         help='HF streaming shuffle buffer; smaller (e.g. 64) reduces cold-start data reads')
    livefed.add_argument('--data-wait-report-seconds',type=float,default=15.,
                         help='Log an in-progress HF batch fetch or model step every N seconds (0 disables)')
    livefed.add_argument('--sync-timeout',type=float,default=30.,help='Max pause to catch up after connection; never indefinite')
    livefed.add_argument('--boot-grace',type=float,default=4.,help='Initial catchup grace, even when all peers are offline')
    livefed.add_argument('--push-fanout',type=int,default=2,choices=range(1,9),metavar='{1..8}',help='How many connected neighbors receive each unsolicited signed update (default 2, lossless NEED repair remains enabled)')
    livefed.add_argument('--inventory-interval',type=float,default=2.5,help='Minimum gap between paged full-history audit listings (signed frontier paging handles fast repair)')
    livefed.add_argument('--inventory-cycle',type=float,default=60.,help='Repeat all-history reconciliation after this many seconds')
    exchange = sub.add_parser('exchange',help='Exchange saved .fpatch via WebRTC')
    exchange.add_argument('--ws', default='ws://127.0.0.1:8080/ws')
    exchange.add_argument('--room', required=True)
    exchange.add_argument('--token', default=os.getenv('APP_TOKEN',''))
    exchange.add_argument('--send', type=Path, help='Previously trained local .fpatch')
    exchange.add_argument('--recv-dir', type=Path, default=Path('received'))
    exchange.add_argument('--timeout', type=float, default=120)
    add_identity(exchange)
    mergep = sub.add_parser('merge',help='Materialize combined adapters into a new HF model')
    add_model_args(mergep)
    mergep.add_argument('--local-patch')
    mergep.add_argument('--remote-patch',action='append',default=[])
    mergep.add_argument('--output-dir',required=True)
    # Explicit full-weight snapshot / transfer operations.
    sexport = sub.add_parser('state-export',help='Serialize current model weights + buffers into data-only .fstate')
    add_model_args(sexport)
    sexport.add_argument('--peer-id',required=True)
    sexport.add_argument('--seq',type=int,default=0)
    sexport.add_argument('--local-patch')
    sexport.add_argument('--remote-patch',action='append',default=[])
    sexport.add_argument('--output',required=True)
    simport = sub.add_parser('state-import',help='Verify and install full state into a model')
    add_model_args(simport)
    simport.add_argument('--state',required=True)
    simport.add_argument('--output-dir',required=True)
    sinspect = sub.add_parser('state-inspect',help='Verify every tensor digest in .fstate')
    sinspect.add_argument('state')
    sinspect.add_argument('--max-state-gb',type=int,default=32)
    sexchange = sub.add_parser('state-exchange',help='Stream full model parameters via WebRTC DataChannel')
    sexchange.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    sexchange.add_argument('--room',required=True)
    sexchange.add_argument('--token',default=os.getenv('APP_TOKEN',''))
    sexchange.add_argument('--send',type=Path,help='Optional full-state archive to send')
    sexchange.add_argument('--recv-dir',type=Path,default=Path('received-states'))
    sexchange.add_argument('--timeout',type=float,default=600)
    sexchange.add_argument('--max-state-gb',type=int,default=32)
    add_identity(sexchange)
    # Tensor-aligned content-addressed shard storage and resumable P2P sync.
    shexport = sub.add_parser('shard-export',help='Stream live model weights directly into shards (no .fstate)')
    add_model_args(shexport)
    shexport.add_argument('--peer-id',required=True)
    shexport.add_argument('--seq',type=int,default=0)
    shexport.add_argument('--local-patch')
    shexport.add_argument('--remote-patch',action='append',default=[])
    shexport.add_argument('--store',type=Path,required=True)
    shexport.add_argument('--output',type=Path,required=True,help='Output .fmanifest')
    shexport.add_argument('--shard-mb',type=int,default=4)
    shcreate = sub.add_parser('shard-create',help='Split existing .fstate into reusable tensor shards')
    shcreate.add_argument('--state',type=Path,required=True)
    shcreate.add_argument('--store',type=Path,required=True)
    shcreate.add_argument('--output',type=Path,required=True,help='Output .fmanifest')
    shcreate.add_argument('--shard-mb',type=int,default=4,help='Each tensor independently split into this many MiB (1-64)')
    shcreate.add_argument('--max-state-gb',type=int,default=32)
    shasm = sub.add_parser('shard-assemble',help='Reconstruct and cryptographically verify .fstate')
    shasm.add_argument('--manifest',type=Path,required=True)
    shasm.add_argument('--store',type=Path,required=True)
    shasm.add_argument('--output',type=Path,required=True)
    shasm.add_argument('--max-state-gb',type=int,default=32)
    shinspect = sub.add_parser('shard-inspect',help='Inspect or verify shards listed in a manifest')
    shinspect.add_argument('--manifest',type=Path,required=True)
    shinspect.add_argument('--store',type=Path,default=Path('shard-store'))
    shinspect.add_argument('--verify',action='store_true')
    shinspect.add_argument('--max-state-gb',type=int,default=32)
    shexchange = sub.add_parser('shard-exchange',help='Request only missing shards over Mummy/WebRTC')
    shexchange.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    shexchange.add_argument('--room',required=True)
    shexchange.add_argument('--token',default=os.getenv('APP_TOKEN',''))
    shexchange.add_argument('--send',type=Path,help='Optional local .fmanifest to advertise')
    shexchange.add_argument('--store',type=Path,required=True,help='Content-addressed shard cache')
    shexchange.add_argument('--recv-dir',type=Path,default=Path('received-manifests'))
    shexchange.add_argument('--assemble-output',type=Path,help='Also reconstruct received .fstate')
    shexchange.add_argument('--send-only',action='store_true',help='Do not expect remote model state')
    shexchange.add_argument('--timeout',type=float,default=3600)
    shexchange.add_argument('--max-state-gb',type=int,default=32)
    add_identity(shexchange)
    real = sub.add_parser('expert-demo-p2p',help='Actual WebRTC direct expert forward+backward demo (all peers start together)')
    real.add_argument('--room-id',required=True)
    real.add_argument('--journal',type=Path,default=Path('expert-demo'))
    real.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    real.add_argument('--token',default=os.getenv('APP_TOKEN',''))
    real.add_argument('--enroll-seconds',type=float,default=8.0)
    real.add_argument('--quant4',action='store_true',help='4-bit frozen expert base and fp32 trainable factors')
    real.add_argument('--kind',choices=['lora','loha'],default='loha')
    real.add_argument('--rank',type=int,default=8)
    sm = sub.add_parser('expert-smoke' ,help='Exercise remote Expert forward+backward in-process')
    scan = sub.add_parser('expert-checkpoint-inspect',help='Read safetensors expert names, no tensor allocation')
    scan.add_argument('--checkpoint',type=Path,required=True)
    discover = sub.add_parser('expert-discover',help='Discover unpacked ModuleList MoE routers in model')
    add_model_args(discover)
    eplan = sub.add_parser('expert-plan',help='Pin weighted rendezvous placements for one stable generation')
    eplan.add_argument('--room-id',required=True)
    eplan.add_argument('--layers',type=Path,required=True,help='JSON list from expert-discover')
    eplan.add_argument('--capacity',type=Path,required=True,help='JSON peer ID -> integer capacity')
    eplan.add_argument('--replicas',type=int,default=2)
    eplan.add_argument('--generation',default='genesis')
    eplan.add_argument('--output',type=Path,required=True)
    code = sub.add_parser('code-register',help='Validate and store a local .py/.zip without executing')
    code.add_argument('--file',type=Path,required=True)
    code.add_argument('--registry',type=Path,default=Path('.fedpara-codes'))
    fetch = sub.add_parser('fetch-custom',help='Download pinned HF repo code+safe weights without executing')
    fetch.add_argument('--repo',required=True)
    fetch.add_argument('--revision',required=True)
    fetch.add_argument('--output-dir',required=True)
    sub.add_parser('room-create',help='Generate a cryptographically random shareable room ID')
    listing=sub.add_parser('room-list',help='List explicitly published server rooms')
    listing.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    info=sub.add_parser('room-info',help='Fetch room model and HF dataset collection without training')
    info.add_argument('--room-id',required=True)
    info.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    info.add_argument('--token',default=os.getenv('APP_TOKEN',''))
    info.add_argument('--output',type=Path,help='Save validated room manifest')
    rm = sub.add_parser('room-manifest-create',help='Create a validated room model/dataset/code descriptor')
    rm.add_argument('--model',required=True)
    rm.add_argument('--revision',default='main')
    rm.add_argument('--dataset',help='Legacy v1 display-only dataset name (use --hf-datasets-file for runnable HF sources)')
    rm.add_argument('--hf-datasets-file',type=Path,help='JSON array of Hugging Face dataset source objects')
    rm.add_argument('--dataset-seed',type=int,default=42)
    rm.add_argument('--description',default='')
    rm.add_argument('--dataset-description',default='')
    rm.add_argument('--license',default='unspecified')
    rm.add_argument('--privacy',choices=['local-only','public','licensed'],default='local-only')
    rm.add_argument('--architecture',default='causal-lm')
    rm.add_argument('--code-file',type=Path)
    rm.add_argument('--code-url',default='')
    rm.add_argument('--output',type=Path,required=True)
    sub.add_parser('gui',help='Embedded HTML/CSS/JS desktop WebView GUI (no HTTP server)')
    sub.add_parser('gui-desktop',help='Legacy native Tkinter GUI')
    keygen = sub.add_parser('keygen',help='Create Ed25519 P2P sender keys')
    keygen.add_argument('--private',type=Path,required=True)
    keygen.add_argument('--public',type=Path,required=True)
    effect=sub.add_parser('benchmark-effectiveness',
        help='Offline real-LoHa: central vs ideal-parallel peer FedSum (NOT WebRTC)')
    effect.add_argument('--model',choices=('toy','tinystories-1m'),default='tinystories-1m',
        help='tinystories-1m=pinned pretrained GPT-Neo; toy=synthetic transformer')
    effect.add_argument('--dataset',choices=('auto','builtin','tinystories'),default='auto',
        help='TinyStories-1M: builtin is a smoke test; tinystories streams actual disjoint HF train/validation examples')
    effect.add_argument('--initialization',choices=('random','pretrained'),default='random',
        help='default random: TinyStories-1M structure initialized from config, frozen base; LoHa only')
    effect.add_argument('--peers',type=int,default=4)
    effect.add_argument('--rounds',type=int,default=5)
    effect.add_argument('--local-steps',type=int,default=2)
    effect.add_argument('--reinit-every',type=int,default=20,
        help='ReLoRA-style merge+reset every N synchronous rounds; 0 disables')
    effect.add_argument('--batch-size',type=int,default=4)
    effect.add_argument('--seed',type=int,default=20261023)
    effect.add_argument('--lr',type=float,default=None,help='auto: 0.0001 for TinyStories-1M; 0.003 for toy')
    effect.add_argument('--kind',choices=('loha','lora'),default='loha')
    effect.add_argument('--optimizer',choices=('adamw','lora-muon'),default='adamw',
        help='LoHa uses AdamW; LoRA-Muon requires --kind lora')
    effect.add_argument('--muon-momentum',type=float,default=.9)
    effect.add_argument('--muon-weight-decay',type=float,default=0.)
    effect.add_argument('--rank',type=int,default=None,
        help='default: 16 for TinyStories-1M (Attention+MLP only); 4 for toy')
    effect.add_argument('--width',type=int,default=24)
    effect.add_argument('--context',type=int,default=None,help='auto: 64 for TinyStories-1M; 12 for toy')
    effect.add_argument('--sequences',type=int,default=None,
        help='training sequences per peer: default 2048 for TinyStories-1M, 64 for toy')
    effect.add_argument('--validation-sequences',type=int,default=None,
        help='heldout validation sequences: default 1024 for TinyStories-1M, 64 for toy')
    effect.add_argument('--warmup-ratio',type=float,default=.05)
    effect.add_argument('--min-lr-ratio',type=float,default=.1)
    effect.add_argument('--grad-clip',type=float,default=1.)
    effect.add_argument('--target-scope',choices=('all','gpt_neo_linear','gpt_neo_full'),default=None)
    effect.add_argument('--threads',type=int,default=1)
    effect.add_argument('--text-file',type=Path,help='Use disjoint local UTF8 byte text corpus instead of synthetic Markov data (requires --vocab 256)')
    effect.add_argument('--vocab',type=int,default=32)
    effect.add_argument('--output-dir',type=Path,default=Path('effectiveness-benchmark'))
    v2bench=sub.add_parser('elastic-v2-benchmark',
        help='8-peer synthetic from-scratch optimizer comparison (in-process, NOT WebRTC)')
    v2bench.add_argument('--peers',type=int,default=8)
    v2bench.add_argument('--epochs',type=int,default=24)
    v2bench.add_argument('--seed',type=int,default=20261022)
    v2bench.add_argument('--local-steps',type=int,default=4)
    v2bench.add_argument('--lr',type=float,default=0.04)
    v2bench.add_argument('--ridge',type=float,default=0.)
    v2bench.add_argument('--output',type=Path,default=Path('elastic-v2-benchmark.json'))
    v2=sub.add_parser('elastic-v2-compare',help='Offline counterfactual V2 aggregation on a completed V1 swarm run (NOT a true V2 training A/B)')
    v2.add_argument('--run',type=Path,required=True,help='Completed Swarm Simulator output directory')
    v2.add_argument('--cap-per-peer',type=int,default=8,help='Up to N earliest events per peer to replay (1..64)')
    v2.add_argument('--eta',type=float,default=1.)
    v2.add_argument('--ridge',type=float,default=0.)
    v2.add_argument('--output',type=Path,help='Optional output JSON; defaults to RUN/elastic-v2-counterfactual.json')
    from .swarm_sim import add_parser as add_swarm_parser
    add_swarm_parser(sub)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    from .mixed_learning_v65 import validate_args as validate_mixed_args
    try:
        validate_mixed_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if getattr(args,'critic_rollout_gossip',False) and not (args.cmd in ('train-live','train-elastic') and args.mode=='elastic'):
        parser.error('--critic-rollout-gossip requires train-elastic or train-live --mode elastic')
    if args.cmd in ('train-live','train-elastic'):
        if getattr(args,'disk_optimizer_dir',None) and (args.mode!='elastic' or args.optimizer!='adamw'):
            parser.error('live disk AdamW requires --mode elastic --optimizer adamw')
        if getattr(args,'disk_optimizer_max_gb',32)<=0:
            parser.error('--disk-optimizer-max-gb must be positive')
        if getattr(args,'resume_disk_optimizer',False):
            parser.error('live disk optimizer restores ONLY its journal-matched working snapshot; omit --resume-disk-optimizer')
        if getattr(args,'streaming_lazy_write',False) and not getattr(args,'streaming_base',False):
            parser.error('--streaming-lazy-write requires --streaming-base')
        if getattr(args,'streaming_lazy_write',False) and getattr(args,'mode',None)!='elastic':
            parser.error('--streaming-lazy-write requires --mode elastic')
        if getattr(args,'stream_lazy_limit_gb',0)<0:
            parser.error('--stream-lazy-limit-gb must be >= 0')
        if getattr(args,'disk_linear_dir',None) and getattr(args,'streaming_base',False):
            parser.error('mutable live disk Linear cannot use the immutable safetensors streaming-base mapping')
        if getattr(args,'live_disk_linear_max_gb',32)<=0:
            parser.error('--live-disk-linear-max-gb must be positive')
        if getattr(args,'auto_rank',False):
            parser.error('P2P adapter rank must match shared genesis: do not use --auto-rank')
        # httpx INFO dumps enormous *signed* CDN redirect URLs for normal 206
        # range responses. Keep errors, but surface actionable batch progress.
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)

    if getattr(args,'hf_shuffle_buffer',512) < 2 or getattr(args,'hf_shuffle_buffer',512) > 100000:
        parser.error('--hf-shuffle-buffer must be in 2..100000')
    if getattr(args,'data_wait_report_seconds',15.0) < 0:
        parser.error('--data-wait-report-seconds cannot be negative')
    if getattr(args,'steps',1) <= 0 or getattr(args,'grad_accum',1) <= 0:
        parser.error('--steps and --grad-accum must be > 0')
    if (args.cmd in ('train','train-p2p','train-live','train-elastic') and
            not any((args.batches_dir,args.text_file,args.jsonl,args.room_manifest)) and
            not (args.cmd in ('train-live','train-elastic') and args.room_id) and
            not (getattr(args,'pt_weight',1.)==0 and
                 (getattr(args,'sft_jsonl',None) or getattr(args,'ppo_jsonl',None) or getattr(args,'sft_hf_repo','') or getattr(args,'ppo_hf_repo','')))):
        parser.error('provide a pretraining source, or set --pt-weight 0 with an SFT/PPO corpus')
    if args.cmd in ('train','train-p2p','train-live','train-elastic') and args.room_manifest and any((args.batches_dir,args.text_file,args.jsonl)):
        from .room_manifest import load_manifest
        if load_manifest(args.room_manifest)['version'] == 2:
            parser.error('HF dataset collection v2 cannot be combined with local --text-file/--jsonl/--batches-dir')
    if args.cmd in ('train','train-p2p') and not args.peer_id:
        parser.error('--peer-id is required for train and train-p2p')
    if args.cmd in ('train-live','train-elastic') and args.mode=='elastic' and not args.room_id and not args.peer_id:
        parser.error('provide --room-id (recommended), or legacy --peer-id and --link')
    if args.cmd=='training-preflight':
        if getattr(args,'pt_weight',1.) == 0:
            parser.error('training-preflight needs a nonzero pretraining objective; use architecture-probe --probe-batch for PPO/SFT-only rooms')
        from .model_contract_v64 import probe_model
        model,adapter=load(args)
        batch=next(training_batches(args))
        target=input_device(model,args.device)
        batch={key:value.to(target) if isinstance(value,torch.Tensor) else value
               for key,value in batch.items()}
        result=probe_model(model,adapter,batch,task=args.hf_task,
                           policy=args.loss_policy,input_map=args.input_map)
        result['target_scope']=args.target_scope
        result['model_type']=str(getattr(getattr(model,'config',None),'model_type',None))
        result['device']=str(target)
        print(json.dumps(result,indent=2,sort_keys=True))
        return
    if args.cmd=='architecture-probe':
        from .model_contract_v64 import probe_model
        model,adapter=load(args)
        batch=load_file(str(args.probe_batch))
        target=input_device(model,args.device)
        batch={key:value.to(target) for key,value in batch.items()}
        result=probe_model(model,adapter,batch,task=args.hf_task,
                           policy=args.loss_policy,input_map=args.input_map)
        result['target_scope']=args.target_scope
        print(json.dumps(result,indent=2,sort_keys=True))
        return
    if args.cmd=='hf-diagnose':
        from .hf_compat_v63 import diagnose_hf_checkpoint
        print(json.dumps(diagnose_hf_checkpoint(args.model,args.revision,local_files_only=args.local_files_only),indent=2))
        return
    if args.cmd=='inspect':
        meta, tensors = decode_patch(Path(args.patch).read_bytes())
        print(json.dumps({**meta,'tensor_count':len(tensors), 'tensor_elements':sum(x.numel() for x in tensors.values())},ensure_ascii=False,indent=2))
    elif args.cmd=='train': train(args)
    elif args.cmd=='train-p2p': asyncio.run(train_p2p(args))
    elif args.cmd in ('train-live','train-elastic'):
        if args.auto_state:
            from .bootstrap import ensure_state
            ensure_state(args, load_model=load_model)
        if args.mode=='synchronous':
            if args.rounds<=0: parser.error('--mode synchronous requires --rounds > 0')
            from .livefed import train_live
            asyncio.run(train_live(args,load_model=load,training_batches=training_batches,run_step=run_step))
        else:
            from .elasticnet import train_elastic
            asyncio.run(train_elastic(args,load_model=load,training_batches=training_batches,run_step=run_step))
    elif args.cmd=='merge': merge(args)
    elif args.cmd=='exchange':
        node = PatchNode(args.ws,args.room,args.token,signer=_signer(args),verifier=_verifier(args))
        path = asyncio.run(exchange_file(node,args.send,args.recv_dir,args.timeout))
        print('received:',path)
    elif args.cmd == 'state-export': state_export_cli(args)
    elif args.cmd == 'state-import': state_import_cli(args)
    elif args.cmd == 'state-inspect':
        meta=verify_state(args.state,max_bytes=args.max_state_gb*1024**3)
        print(json.dumps({k:v for k,v in meta.items() if k != 'tensors'} | {'tensor_count':len(meta['tensors'])},indent=2))
    elif args.cmd == 'state-exchange': state_exchange_cli(args)
    elif args.cmd == 'shard-create': shard_create_cli(args)
    elif args.cmd == 'shard-export': shard_export_cli(args)
    elif args.cmd == 'shard-assemble': shard_assemble_cli(args)
    elif args.cmd == 'shard-inspect': shard_inspect_cli(args)
    elif args.cmd == 'shard-exchange': shard_exchange_cli(args)
    elif args.cmd=='expert-demo-p2p':
        from .expert_demo import run_expert_demo
        asyncio.run(run_expert_demo(args))
    elif args.cmd=='expert-smoke': expert_smoke_cli(args)
    elif args.cmd=='expert-checkpoint-inspect': expert_checkpoint_inspect(args)
    elif args.cmd=='expert-plan': expert_plan_cli(args)
    elif args.cmd=='expert-discover': expert_discover_cli(args)
    elif args.cmd=='code-register': code_register_cli(args)
    elif args.cmd == 'room-manifest-create':
        from .room_manifest import make_manifest,make_hf_manifest,manifest_digest
        if args.hf_datasets_file:
            sources=json.loads(args.hf_datasets_file.read_text(encoding='utf8'))
            manifest=make_hf_manifest(args.model,args.revision,sources,seed=args.dataset_seed,
                 architecture=args.architecture,description=args.description,
                 code_path=args.code_file,code_url=args.code_url)
        else:
            if not args.dataset: parser.error('provide --dataset or --hf-datasets-file')
            manifest=make_manifest(args.model,args.revision,args.dataset,
                architecture=args.architecture,description=args.description,
                dataset_description=args.dataset_description,license=args.license,
                privacy=args.privacy,code_path=args.code_file,code_url=args.code_url)
        args.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print('manifest:',args.output,'sha256:',manifest_digest(manifest))
    elif args.cmd == 'gui':
        from .webview_gui import run_webview
        run_webview()
    elif args.cmd == 'gui-desktop':
        from .gui import run_desktop
        run_desktop()
    elif args.cmd == 'room-info':
        from .roomid import fetch_room_manifest
        from .room_manifest import manifest_digest
        m=asyncio.run(fetch_room_manifest(args.ws,args.room_id,args.token))
        if args.output:
            args.output.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'manifest':m,'sha256':manifest_digest(m)},indent=2,ensure_ascii=False))
    elif args.cmd == 'room-list':
        from .roomid import list_public_rooms
        print(json.dumps(list_public_rooms(args.ws),ensure_ascii=False,indent=2))
    elif args.cmd == 'benchmark-effectiveness':
        from .effectiveness_benchmark import run_effectiveness_benchmark
        result=run_effectiveness_benchmark(peers=args.peers,rounds=args.rounds,
            local_steps=args.local_steps,batch_size=args.batch_size,seed=args.seed,
            lr=args.lr,kind=args.kind,rank=args.rank,width=args.width,
            context=(args.context if args.context is not None else (12 if args.model=='toy' else 64)),
            sequences=args.sequences,validation_sequences=args.validation_sequences,
            threads=args.threads,
            text_file=args.text_file,vocab=args.vocab,model=args.model,dataset=args.dataset,
            warmup_ratio=args.warmup_ratio,min_lr_ratio=args.min_lr_ratio,
            grad_clip=args.grad_clip,target_scope=args.target_scope,initialization=args.initialization,
            optimizer=args.optimizer,muon_momentum=args.muon_momentum,
            muon_weight_decay=args.muon_weight_decay,
            reinit_every=args.reinit_every,output_dir=args.output_dir)
        print(json.dumps({'status':result['status'],'loss':result['loss'],
            'compute':result['compute'],'artifacts':result['artifacts']},
            ensure_ascii=False,indent=2),flush=True)
    elif args.cmd == 'elastic-v2-benchmark':
        from .elastic_v2_benchmark import run_optimizer_benchmark
        result=run_optimizer_benchmark(peers=args.peers,epochs=args.epochs,
                    seed=args.seed,local_steps=args.local_steps,lr=args.lr,ridge=args.ridge)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('status','loss_first','loss_last','optimizer_time_s','warning')},indent=2))
        print('Saved:',args.output)
    elif args.cmd == 'elastic-v2-compare':
        from .elastic_v2 import compare_legacy_swarm_trace
        if not 1 <= args.cap_per_peer <= 64:
            parser.error('--cap-per-peer must be in 1..64')
        result=compare_legacy_swarm_trace(args.run,cap_per_peer=args.cap_per_peer,
                                          eta=args.eta,ridge=args.ridge)
        output=args.output or args.run/'elastic-v2-counterfactual.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        print('Counterfactual analysis saved:',output)
    elif args.cmd == 'swarm-sim':
        from .swarm_sim import cli as run_swarm_sim
        run_swarm_sim(args)
    elif args.cmd == 'room-create':
        from .roomid import create_room_id
        print(create_room_id())
    elif args.cmd == 'keygen': generate_keypair(args.private,args.public)
    elif args.cmd == 'fetch-custom': fetch_custom_cli(args)


if __name__ == '__main__':
    main()
