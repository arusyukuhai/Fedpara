"""Kagami local inference with *verified learned effective weights*.

No remote/factory Python code is executed. For a trained checkpoint we construct
only the trusted built-in Transformers model architecture, then directly install
its immutable, SHA-256 verified state shards before generation. Quantization
of trained checkpoints uses a temporary safetensors export (disk intensive).
This is a subprocess, not a security isolation boundary for untrusted model code.
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path


QUANTIZED=('int8','fp4','nf4','torchao-int4')
PRECISIONS=('auto','fp32','fp16','bf16',*QUANTIZED)


def resolve_precision(torch, precision):
    """Select a precision actually supported by this device, not a CUDA default."""
    if precision not in PRECISIONS:
        raise ValueError('unsupported inference precision: '+str(precision))
    if precision != 'auto':
        return precision
    if torch.cuda.is_available():
        return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    mps=getattr(torch.backends,'mps',None)
    if mps is not None and mps.is_available():
        return 'fp16'
    return 'fp32'


def _compute_device(torch):
    if torch.cuda.is_available():
        return 'cuda'
    mps=getattr(torch.backends,'mps',None)
    if mps is not None and mps.is_available():
        return 'mps'
    return 'cpu'


def _context_window(model, tokenizer, requested_tokens):
    """Reserve output tokens inside a real finite model context window.

    A tokenizer's default model_max_length can be a giant sentinel; ignore it.
    """
    config=getattr(model,'config',None)
    bounds=[]
    for attr in ('max_position_embeddings','n_positions','max_sequence_length','seq_length'):
        value=getattr(config,attr,None)
        if isinstance(value,int) and not isinstance(value,bool) and 1 < value < 10_000_000:
            bounds.append(value)
    token_limit=getattr(tokenizer,'model_max_length',None)
    if isinstance(token_limit,int) and not isinstance(token_limit,bool) and 1 < token_limit < 10_000_000:
        bounds.append(token_limit)
    # Keep established upper bound to prevent huge prompts exhausting memory.
    total=min([2048,*bounds])
    if total <= 1:
        raise ValueError('モデルの文脈長が短すぎて推論できません')
    effective_tokens=min(requested_tokens, max(1,total-1))
    if requested_tokens>=total:
        # Reserve half the window for the prompt, instead of failing on the
        # default MAX NEW TOKENS when a small custom model has context=128.
        effective_tokens=max(1,min(requested_tokens,total//2))
    return total-effective_tokens,effective_tokens


def friendly_error(exc):
    """Supplement the original exception with an actionable, non-misleading hint."""
    message=str(exc)
    lower=message.lower()
    if isinstance(exc,ModuleNotFoundError):
        return '必要な推論ライブラリがありません。transformers / accelerate / safetensors、量子化の場合は対応する追加ライブラリを導入してください。'
    if 'checkpoint' in lower or 'shard' in lower or 'snapshot' in lower:
        return '学習済みモデルの保存状態を確認してください。破損・構造不一致の場合は同じモデルのスナップショットを再保存してください。'
    if any(x in lower for x in ('tokenizer','vocab','embedding','input_ids')):
        return 'モデルとTokenizerの組み合わせ・語彙サイズを確認してください。'
    if any(x in lower for x in ('cuda','bitsandbytes','quantiz','torchao','fp16','bf16','device_map')):
        return '指定した精度・量子化方式がこの端末に対応しているか確認してください。「AUTO」か「FP32」でも試してください。'
    if any(x in lower for x in ('position','sequence length','maximum length','max_length','context')):
        return '入力文がモデルの文脈長を超えています。入力または生成トークン数を減らしてください。'
    if any(x in lower for x in ('repo','revision','pretrained','config')):
        return 'Hugging FaceのモデルID、revision、アーキテクチャ設定を確認してください。'
    return '詳細はEVALのエラーログを確認してください。'



def _transformers_dtype_key():
    """Transformers 4.48 supports torch_dtype; newer releases prefer dtype."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        from re import match
        raw=version('transformers')
        found=match(r'^(\d+)\.(\d+)', raw)
        if found and tuple(map(int, found.groups())) < (4, 56):
            return 'torch_dtype'
    except (ImportError, ValueError, PackageNotFoundError):
        pass
    return 'dtype'


def model_options(torch, precision):
    cuda=torch.cuda.is_available()
    device=_compute_device(torch)
    opts={'device_map':'auto' if device!='cpu' else 'cpu','trust_remote_code':False,
          'use_safetensors':True}
    if precision in ('fp4','nf4','int8'):
        if not cuda: raise RuntimeError('bitsandbytes quantization requires supported hardware/backend')
        from transformers import BitsAndBytesConfig
        if precision=='int8': opts['quantization_config']=BitsAndBytesConfig(load_in_8bit=True)
        else:
            opts['quantization_config']=BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type=precision,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16))
    elif precision=='torchao-int4':
        from transformers import TorchAoConfig
        from torchao.quantization import Int4WeightOnlyConfig
        opts['quantization_config']=TorchAoConfig(quant_type=Int4WeightOnlyConfig(group_size=128))
    else:
        if precision=='fp16' and device=='cpu':
            raise ValueError('FP16 inference on CPU is unsupported; select AUTO (FP32 on CPU) or FP32')
        opts[_transformers_dtype_key()]={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[precision]
    return opts


def _model_class(manifest):
    # Only trusted Transformers AutoModel factories. No network-supplied code.
    import transformers
    task=manifest['model']['architecture']
    names={'causal-lm':'AutoModelForCausalLM',
           'seq2seq-lm':'AutoModelForSeq2SeqLM',
           'masked-lm':'AutoModelForMaskedLM',
           'sequence-classification':'AutoModelForSequenceClassification'}
    if task not in names:
        raise ValueError('Unsupported evaluation task: '+str(task))
    return getattr(transformers,names[task])


def load_snapshot_model(options, torch, manifest):
    """Load exact effective P2P model weights, never infer from HF base."""
    from .hf_config_v83 import load_hf_config
    model_cls=_model_class(manifest)
    from .model_design import create_config
    from .shards import read_manifest, apply_sharded_state

    snap=Path(options['checkpoint']).resolve(strict=True)
    store=Path(options['store']).resolve(strict=True)
    if not snap.is_file() or not store.is_dir():
        raise ValueError('snapshot or shard store not found')
    state=read_manifest(snap,max_state_bytes=2048*1024**3)
    if state['model_id']!=manifest['model']['id'] or state['revision']!=options['checkpoint_revision']:
        raise ValueError('checkpoint identity mismatch')
    if state['total_tensor_bytes'] <= 0:
        raise ValueError('checkpoint is empty')
    config=(create_config(manifest) if 'design' in manifest else
            load_hf_config(manifest['model']['id'],
               revision=manifest['model']['revision'],trust_remote_code=False))
    # Creating on meta avoids allocating random parameters just to overwrite.
    with torch.device('meta'):
        model=model_cls.from_config(config,trust_remote_code=False)
    model.to_empty(device='cpu')
    # Restore in the *checkpoint's* original dtype, not the factory default.
    # This preserves a 16-bit learned model even when AutoConfig defaults to fp32.
    saved_dtype=_direct_restore_dtype(state)
    if saved_dtype is not None:
        model.to(dtype=getattr(torch,saved_dtype.removeprefix('torch.')))
    model.tie_weights()
    apply_sharded_state(model,snap,store,model_id=state['model_id'],
        revision=state['revision'],max_state_bytes=2048*1024**3)
    model.eval()
    return model,state


def _snapshot_fits_cpu(state):
    """Conservatively choose in-memory restore for small checkpoints only."""
    try:
        import psutil
        free=psutil.virtual_memory().available
        # to_empty + state installation + dtype conversion; leave plenty of RAM.
        return state['total_tensor_bytes'] * 3 < free * .50
    except ImportError:
        return False


def _direct_restore_dtype(state):
    """Return the single trained floating dtype, or None for mixed precision.

    ``to_empty`` uses the HF config dtype (often float32) rather than the
    checkpoint dtype. Loading a fp16/bf16 checkpoint into float32 destinations
    previously raised ``sharded snapshot destination shape/dtype mismatch``.
    Mixed-precision snapshots are streamed through safetensors instead: casting
    all trainable and nontrainable tensors to one dtype would corrupt fidelity.
    """
    kinds={spec['dtype'] for spec in state['tensors'] if spec['dtype'] in
           ('torch.float16','torch.bfloat16','torch.float32','torch.float64')}
    return next(iter(kinds)) if len(kinds)==1 else None


def _direct_restore_eligible(state):
    if _direct_restore_dtype(state) is None:
        return False
    return _snapshot_fits_cpu(state)


def _expected_weights_match(torch, model_cls, config, state):
    """Never silently initialize omitted HF tensors in place of learned ones."""
    from .core import PatchError
    with torch.device('meta'):
        shell=model_cls.from_config(config,trust_remote_code=False)
    expected=shell.state_dict()
    provided={t['name']:t for t in state['tensors']}
    if set(expected)!=set(provided):
        missing=list(set(expected)-set(provided))[:5]
        extra=list(set(provided)-set(expected))[:5]
        raise PatchError(f'model checkpoint keys differ: missing={missing} extra={extra}')
    for name,tensor in expected.items():
        if tuple(tensor.shape)!=tuple(provided[name]['shape']):
            raise PatchError('trained tensor shape incompatible with HF architecture: '+name)
    del shell


from contextlib import contextmanager

@contextmanager
def _streaming_snapshot_model(options, torch, manifest, config, precision, state):
    """Reconstruct via disk-streamed HF safetensors + Accelerate auto dispatch."""
    model_cls=_model_class(manifest)
    from .snapshot_safetensors import export_hf_safetensors
    from shutil import disk_usage

    _expected_weights_match(torch,model_cls,config,state)
    config_args=model_options(torch,precision)  # Reject unsupported quantizers before exporting huge weights.
    temp_root=Path(options['temp_dir']).resolve(strict=True)
    # Full-size staging can be unavoidable with HF quantizers; surface this
    # before export rather than caching a giant half-checkpoint.
    needed=int(state['total_tensor_bytes']*1.05)+100*1024**2
    if disk_usage(temp_root).free<needed:
        raise OSError('insufficient temporary disk for streaming learned checkpoint')
    checkpoint=Path(options['checkpoint']).resolve(strict=True)
    store=Path(options['store']).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='kagami-hf-',dir=temp_root) as root:
        destination=Path(root)/'weights'
        print('Kagami: exporting SHA256-verified shards to HF safetensors with bounded RAM',flush=True)
        result=export_hf_safetensors(checkpoint,store,destination,
                                    max_state_bytes=2048*1024**3)
        if (result['model_id'],result['revision'])!=(state['model_id'],state['revision']):
            raise ValueError('checkpoint identity changed during streaming export')
        from .hf_config_v83 import save_hf_config_with_model_type
        save_hf_config_with_model_type(config, destination)
        config_args['device_map']='auto'
        config_args['low_cpu_mem_usage']=True
        config_args['offload_folder']=str(Path(root)/'offload')
        print(f'Kagami: loading {result["files"]} safetensors with Accelerate auto dispatch',flush=True)
        model=model_cls.from_pretrained(str(destination),**config_args)
        # Accelerate dispatch may rely on on-disk offloaded weights. Inference
        # must finish BEFORE the temporary offload folder is removed.
        yield model


def run(options):
    import torch
    from .hf_load_v85 import load_hf_tokenizer
    from .room_manifest import validate_manifest
    precision=resolve_precision(torch, options.get('precision','auto'))
    manifest=validate_manifest(options['manifest'])
    task=manifest['model']['architecture']
    model_cls=_model_class(manifest)
    source=options.get('scope','trained')
    if source not in ('trained','base'):
        raise ValueError('bad inference scope')
    name=manifest['model']['id'];revision=manifest['model']['revision']
    from .model_design import tokenizer_identity
    tokenizer_id,tokenizer_revision=tokenizer_identity(manifest)
    tok=load_hf_tokenizer(tokenizer_id,revision=tokenizer_revision,trust_remote_code=False)
    state=None
    from contextlib import contextmanager
    @contextmanager
    def selected_model():
        nonlocal state
        if source=='trained':
            from .model_design import create_config
            from .shards import read_manifest
            from .hf_config_v83 import load_hf_config
            state=read_manifest(Path(options['checkpoint']),max_state_bytes=2048*1024**3)
            if ((state['model_id'],state['revision']) !=
                (manifest['model']['id'],options['checkpoint_revision'])):
                raise ValueError('trained snapshot model identity mismatch')
            if state['revision']!=manifest['model']['revision']:
                raise ValueError('learned snapshot revision differs from selected model')
            config=(create_config(manifest) if 'design' in manifest else
                    load_hf_config(name,revision=revision,trust_remote_code=False))
            config.use_cache=True
            if precision not in QUANTIZED and _direct_restore_eligible(state):
                print('Kagami: restoring verified learned weights in RAM',flush=True)
                model=None
                try:
                    model,_=load_snapshot_model(options,torch,manifest)
                    opts=model_options(torch,precision)
                    model=model.to(dtype=opts.get('dtype',opts.get('torch_dtype')))
                    target=_compute_device(torch)
                    if target!='cpu':
                        model=model.to(target)
                except (torch.OutOfMemoryError, MemoryError):
                    # Recover gracefully when a small-on-RAM checkpoint does not
                    # actually fit in VRAM. The streaming fallback uses
                    # Accelerate's real device dispatch and CPU/disk offload.
                    print('Kagami: direct restore exhausted memory, retrying with streaming offload',flush=True)
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                    with _streaming_snapshot_model(options,torch,manifest,config,precision,state) as streamed:
                        yield streamed
                else:
                    try:yield model
                    finally:del model
            else:
                print('Kagami: bounded-RAM streaming/offloaded EVAL selected',flush=True)
                with _streaming_snapshot_model(options,torch,manifest,config,precision,state) as model:
                    yield model
        else:
            if 'design' in manifest:
                raise ValueError('Newly initialized models have no pretrained HF base; save a checkpoint first')
            print('Kagami: explicitly loading original HF base weights',flush=True)
            from .hf_config_v83 import load_hf_config
            config=load_hf_config(name,revision=revision,trust_remote_code=False)
            model=model_cls.from_pretrained(name,revision=revision,config=config,
                                  **model_options(torch,precision))
            try:yield model
            finally:del model

    with selected_model() as model:
        model.eval()
        # Classification and masked prediction do not create decoder tokens.
        new_token_budget=options['max_tokens'] if task in ('causal-lm','seq2seq-lm') else 0
        usable_context,effective_tokens=_context_window(model,tok,new_token_budget)
        if new_token_budget and effective_tokens!=new_token_budget:
            print(f'Kagami: capped generation from {new_token_budget} to {effective_tokens} tokens for this context',flush=True)
        inputs=tok(options['prompt'],return_tensors='pt',truncation=True,max_length=usable_context)
        if inputs['input_ids'].shape[-1]==0:
            starter=tok.bos_token_id if getattr(tok,'bos_token_id',None) is not None else tok.eos_token_id
            if starter is None:
                raise ValueError('空の入力にはBOSまたはEOSトークンが必要です。プロンプトを入力してください。')
            inputs={'input_ids':torch.tensor([[starter]],dtype=torch.long),
                    'attention_mask':torch.ones((1,1),dtype=torch.long)}
        emb=model.get_input_embeddings()
        device=next(emb.parameters()).device if emb is not None else next(model.parameters()).device
        if device.type=='meta':
            # Accelerate may offload embedding weights to disk and materialize
            # them only via a forward pre-hook; do not send input IDs to meta.
            # CPU IDs are safe inputs for its dispatch hooks.
            device=torch.device('cpu')
        inputs={k:v.to(device) for k,v in inputs.items()}
        pad_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if pad_id is None:
            pad_id=getattr(model.config,'pad_token_id',None)
        with torch.inference_mode():
            if task in ('causal-lm','seq2seq-lm'):
                if pad_id is None:
                    raise ValueError('Tokenizer/model has neither pad_token_id nor eos_token_id; cannot generate safely')
                generated=model.generate(**inputs,max_new_tokens=effective_tokens,
                                         do_sample=False,pad_token_id=pad_id)
                # Encoder-decoder generate() does not prefix the input prompt.
                new_ids=(generated[0][inputs['input_ids'].shape[-1]:]
                         if task=='causal-lm' else generated[0])
                text=tok.decode(new_ids,skip_special_tokens=True)
                generated_tokens=int(new_ids.numel())
            elif task=='sequence-classification':
                logits=model(**inputs).logits
                probabilities=logits[0].float().softmax(-1)
                index=int(probabilities.argmax().item())
                label=getattr(model.config,'id2label',{}).get(index,str(index))
                score=float(probabilities[index].item())
                text=f'{label} ({score:.2%})'
                generated_tokens=0
            else:
                # Masked language modeling: predict each [MASK] token's top-1.
                mask_id=getattr(tok,'mask_token_id',None)
                if mask_id is None:
                    raise ValueError('Tokenizer has no [MASK] token; use a compatible masked-LM tokenizer')
                mask_positions=(inputs['input_ids'][0]==mask_id).nonzero(as_tuple=False).flatten()
                if mask_positions.numel()==0:
                    raise ValueError('Masked LM requires a [MASK] token in the prompt')
                logits=model(**inputs).logits[0]
                replaced=inputs['input_ids'][0].clone()
                for position in mask_positions:
                    replaced[position]=logits[position].argmax(-1)
                text=tok.decode(replaced,skip_special_tokens=True)
                generated_tokens=int(mask_positions.numel())
        return {'text':text,'model':name,'task':task,
                'precision':precision,'tokens':generated_tokens,
                'requested_max_tokens':options['max_tokens'],
                'effective_max_tokens':effective_tokens,
                'scope':('Kagami learned effective checkpoint' if state else 'Hugging Face base (explicit)'),
                'checkpoint_seq':state['seq'] if state else None,
                'checkpoint_tensors':len(state['tensors']) if state else None}

def main():
    input_path=Path(sys.argv[1]);output=Path(sys.argv[2])
    try:
        payload=json.loads(input_path.read_text(encoding='utf8'))
        result=run(payload)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result={'error':type(exc).__name__+': '+str(exc)[:1200],
                'hint':friendly_error(exc), 'error_type':type(exc).__name__}
    finally:
        input_path.unlink(missing_ok=True)
    tmp=output.with_suffix('.tmp')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as fh:
        json.dump(result,fh,ensure_ascii=False)
        fh.flush();os.fsync(fh.fileno())
    os.replace(tmp,output)

if __name__=='__main__':main()
