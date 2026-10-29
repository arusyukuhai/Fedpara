"""Safe, built-in architecture preflight for existing Hugging Face models.

Do not execute remote code inside Kagami P2P peers.  This intentionally
rejects FP8 native DeepSeek checkpoints until their fused kernels and
quantization conventions can be verified in the training engine.
"""
from __future__ import annotations

def validate_training_config(config, model_id, *, streaming=False):
    model_type=str(getattr(config,'model_type','') or '').lower().replace('-','_')
    if not model_type:
        raise ValueError('Missing model_type: Hugging Face config.json must declare the model architecture')
    quant=getattr(config,'quantization_config',None)
    name=str(model_id).lower()
    # Architecture, not merely repository brand, decides compatibility.
    # R1-Distill-Qwen and R1-Distill-Llama use conventional HF configs.
    # Do not reject a model based on architecture name: modern installed
    # Transformers may supply a compatible differentiable implementation.
    # The actual loader + model contract determines support. Native FP8 and
    # quantized kernels still require a checked dequantization backend.
    if quant:
        method=(quant.get('quant_method') if isinstance(quant,dict) else getattr(quant,'quant_method','unknown'))
        raise ValueError(f'Kagami does not have an audited trainable backend for {method} quantized tensors. '
                         'Supply non-quantized BF16/FP16 safetensors and LoRA/LoHA adapters.')
    # Avoid declaring all unknown architectures supported; the built-in
    # transformers factory/adapter then performs the concrete compatibility test.
    return model_type


def diagnose_hf_checkpoint(model_id, revision='main', *, local_files_only=False):
    """Inspect config and checkpoint presence without executing model code.

    Returns a JSON-compatible report.  A positive report is not a GPU/RAM,
    architecture-kernel, or full-training certification.
    """
    import json
    from pathlib import Path
    from types import SimpleNamespace
    path=Path(model_id).expanduser()
    if path.is_dir():
        config_path=path/'config.json'
        source=path
    else:
        try:
            from huggingface_hub import hf_hub_download
            config_path=Path(hf_hub_download(model_id,'config.json',revision=revision,
                                             local_files_only=local_files_only))
        except Exception as exc:
            typ=type(exc).__name__
            if 'Gated' in typ or '401' in str(exc):
                raise PermissionError(f'Access denied to {model_id}. Accept the model license on Hugging Face '
                                      'and authenticate locally with `hf auth login`.') from exc
            raise RuntimeError(f'Unable to inspect {model_id}@{revision}: {typ}: {exc}') from exc
        source=None
    if not config_path.is_file():
        raise FileNotFoundError(f'No config.json in {model_id}')
    cfg=json.loads(config_path.read_text(encoding='utf8'))
    if not isinstance(cfg,dict):
        raise ValueError('invalid Hugging Face config.json')
    from .hf_config_v83 import checked_model_type, MissingModelTypeError
    identity_error = None
    inferred = False
    try:
        detected_type = checked_model_type(cfg, source=str(model_id))
        inferred = not bool(cfg.get('model_type'))
    except MissingModelTypeError as exc:
        detected_type = None
        identity_error = str(exc)
    report={'model':str(model_id), 'revision':revision,
            'model_type':detected_type, 'architectures':cfg.get('architectures'),
            'checkpoint_present':False, 'streaming_candidate':False,
            'notes':[]}
    if source is not None:
        report['checkpoint_present']=bool((source/'model.safetensors').is_file() or
                                          (source/'model.safetensors.index.json').is_file())
        report['local_safetensors_bytes']=sum(p.stat().st_size for p in source.glob('*.safetensors'))
        if not report['checkpoint_present']:
            report['notes'].append('Expected model.safetensors or sharded model.safetensors.index.json; .bin/pickle not supported in safe streaming mode')
    if identity_error:
        report['notes'].append(identity_error)
        return report
    if inferred:
        report['notes'].append('model_type inferred from unique built-in architectures; Kagami can normalize it on export without modifying the original')
    try:
        validate_training_config(SimpleNamespace(**(cfg | {'model_type':detected_type})),model_id,streaming=True)
    except ValueError as exc:
        report['notes'].append(str(exc))
    else:
        report['streaming_candidate']=True
        report['notes'].append('Config is a possible built-in Transformers candidate; verify with real loading, tokenizer, autograd and available device resources')
    return report
