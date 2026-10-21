"""Local immutable sandbox input registry for user-supplied .py or .zip factories.

HTTP upload never invokes code. Other peers get a digest, not executable code.
No pip installation or Python import occurs on registration.
"""
from __future__ import annotations
import base64
import hashlib
import io
import os
import re
import stat
import zipfile
from pathlib import Path

MAX_CODE=256*1024
MAX_FILES=40
SAFE_PART=re.compile(r'^[A-Za-z0-9_.-]{1,80}$')


def validate_source(blob:bytes,name:str)->tuple[str,str]:
    if not isinstance(blob,bytes) or not 0<len(blob)<=MAX_CODE:raise ValueError('code size must be 1..256KiB')
    if not isinstance(name,str) or not SAFE_PART.fullmatch(name):raise ValueError('invalid code filename')
    if name.endswith('.py'):
        blob.decode('utf-8')
        return hashlib.sha256(blob).hexdigest(),'python'
    if not name.endswith('.zip'):raise ValueError('upload only .py or .zip')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        entries=z.infolist()
        if not 1<=len(entries)<=MAX_FILES:raise ValueError('zip file count must be 1..40')
        total=0
        for info in entries:
            n=info.filename
            parts=n.rstrip('/').split('/')
            if (n.startswith('/') or '..' in parts or any(not SAFE_PART.fullmatch(p) for p in parts) or
                '\\' in n or info.flag_bits & 1):
                raise ValueError('invalid archive path or encrypted member')
            mode=info.external_attr>>16
            if stat.S_ISLNK(mode):raise ValueError('archive symlinks forbidden')
            if info.is_dir():continue
            if not n.endswith('.py') and n!='README.md':
                raise ValueError('only .py and README.md permitted in model source ZIP')
            total+=info.file_size
            if info.file_size>MAX_CODE or total>MAX_CODE:raise ValueError('expanded source too large')
            z.read(info)  # CRC validation; no extraction
    return hashlib.sha256(blob).hexdigest(),'python-zip'


def register_code(root:Path,name:str,blob:bytes)->dict:
    digest,kind=validate_source(blob,name)
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(root,0o700)
    ext='.py' if kind=='python' else '.zip'
    dest=root/(digest+ext)
    if not dest.exists():
        fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            with os.fdopen(fd,'wb') as fp:fp.write(blob);fp.flush();os.fsync(fp.fileno())
        except BaseException:
            dest.unlink(missing_ok=True);raise
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:
        raise PermissionError('content-addressed code mismatch')
    return {'sha256':digest,'kind':kind,'path':str(dest),'size':len(blob)}


def register_base64(root:Path,name:str,contents:str)->dict:
    if not isinstance(contents,str) or len(contents)>360_000:raise ValueError('upload too large')
    return register_code(root,name,base64.b64decode(contents,validate=True))
