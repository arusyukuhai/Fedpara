#!/usr/bin/env python3
"""Explicit local repair of a legacy Hugging Face config.json.

Usage:
  python scripts/fix_hf_model_type_v83.py /path/to/model
  python scripts/fix_hf_model_type_v83.py /path/to/model --write
  python scripts/fix_hf_model_type_v83.py /path/to/model --model-type gpt2 --write

This edits *only* local config.json and creates config.json.before-v83.bak once.
Do not change signed / SHA256-pinned checkpoints: a changed file invalidates
its original integrity commitment and should require a new revision/genesis.
"""
import argparse
import json
import os
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchlib.hf_config_v83 import checked_model_type, MissingModelTypeError, _BUILTIN_ARCHITECTURES


def repair_local(path: Path, *, model_type: str | None=None, write=False) -> str:
    path=Path(path).expanduser().resolve()
    config_path=path/'config.json' if path.is_dir() else path
    if config_path.name!='config.json' or not config_path.is_file():
        raise FileNotFoundError('Specify a local Hugging Face model folder or config.json')
    data=json.loads(config_path.read_text(encoding='utf8'))
    if not isinstance(data,dict):
        raise ValueError('config.json must be an object')
    declared=data.get('model_type')
    if declared is not None:
        actual=checked_model_type(data, source=str(config_path))
        if model_type is not None and model_type!=actual:
            raise ValueError('Refusing to overwrite an existing different model_type')
    elif model_type is not None:
        # Explicit architecture selection must remain one of the built-in
        # audited types for this repair tool (not arbitrary custom code).
        if model_type not in _BUILTIN_ARCHITECTURES.values():
            raise ValueError('Unrecognized built-in --model-type; choose a supported HF architecture')
        try:
            inferred=checked_model_type(data,source=str(config_path))
        except MissingModelTypeError:
            inferred=None
        if inferred is not None and inferred!=model_type:
            raise ValueError('Chosen model_type disagrees with declared built-in architectures')
        actual=model_type
    else:
        actual=checked_model_type(data,source=str(config_path))
    if write and declared is None:
        backup=config_path.with_name('config.json.before-v83.bak')
        if backup.exists():
            raise FileExistsError('Repair backup already exists; refusing a second edit')
        # The backup represents the exact original bytes, not JSON-reformatted.
        shutil.copy2(config_path,backup)
        data['model_type']=actual
        temp=config_path.with_name('.config-'+secrets.token_hex(8)+'.tmp')
        try:
            with temp.open('x',encoding='utf8') as output:
                json.dump(data,output,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
                output.write('\n');output.flush();os.fsync(output.fileno())
            os.replace(temp,config_path)
        finally:
            temp.unlink(missing_ok=True)
    return actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--model-type',help='Explicit builtin HF architecture (e.g. gpt2, llama)')
    parser.add_argument('--write',action='store_true',help='Edit local config.json with one-time backup')
    args=parser.parse_args()
    try:
        resolved=repair_local(args.folder,model_type=args.model_type,write=args.write)
    except (OSError,ValueError,json.JSONDecodeError) as exc:
        parser.exit(2,f'Cannot repair configuration: {exc}\n')
    print(f'model_type={resolved} ({"written" if args.write else "dry run"})')


if __name__=='__main__':
    main()
