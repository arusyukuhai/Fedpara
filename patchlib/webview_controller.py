"""Business logic behind Fedpara's embedded WebView desktop UI.

No HTTP listener is created. It does not execute custom model code.
Room manifests only contain validated, hash-pinned metadata.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import shutil
import subprocess
import sys
import datetime
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .gui import DesktopController
from .room_manifest import (make_hf_manifest, manifest_digest, validate_manifest,
                            validate_hf_sources, load_manifest)
from .roomid import create_room_id, validate_room, fetch_room_manifest, list_public_rooms, register_room_manifest
from .profiles import allocate_profile


def check_signaling_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 512:
        raise ValueError('invalid signaling URL')
    parts = urlsplit(url)
    if (parts.scheme not in ('ws','wss') or not parts.hostname or
        parts.username or parts.password or parts.fragment):
        raise ValueError('invalid signaling URL')
    if parts.scheme == 'ws' and parts.hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('remote signaling requires wss://')
    return url

@contextmanager
def _exclusive_manifest_lock(lockfile: Path):
    """Serialize simultaneous GUI saves of the same room across processes.

    On macOS/Linux the lock belongs to the OS file descriptor and is released
    even if a window crashes. Do not erase this small lock file while running.
    """
    import fcntl
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class WebviewController:
    def __init__(self, workdir: Path | None = None):
        self.workdir = (Path(workdir) if workdir else Path.cwd()).resolve()
        self.train = DesktopController()
        self.lock = threading.RLock()
        self.imported: dict[str, dict] = {}
        self.profiles = {}  # one stable, exclusive profile per GUI instance / room
        self._bench_proc = None
        self._bench_log = None
        self._bench_output = None


    def _room_path(self, room: str) -> Path:
        validate_room(room)
        return self.workdir / '.fedpara-gui' / 'rooms' / room / 'manifest.json'

    def settings(self):
        return {'room':create_room_id(), 'model':'roneneldan/TinyStories-1M',
                'revision':'77f1b168e219585646439073245fe87e56b3023e', 'task':'causal-lm', 'ws':'ws://127.0.0.1:8080/ws',
                'seed':42,'kind':'loha','rank':16,'initialization':'random',
                'reinit_every':20,'lr':0.0001,'local_steps':64,'max_length':64,
                'sources':[{'repo':'roneneldan/TinyStories','revision':'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64','config':None,
                            'split':'train','columns':{'text':'text'},'weight':1.0}]}

    def save(self, payload: dict):
        room = validate_room(payload['room'])
        source = validate_hf_sources(payload['sources'])
        model, revision, task = payload['model'],payload['revision'],payload['task']
        from .live_initialization import (room_initialization,
                                          RANDOM_MANIFEST_DESCRIPTION)
        initialization=payload.get('initialization',
                                   'random' if model=='roneneldan/TinyStories-1M' else 'pretrained')
        if initialization not in ('random','pretrained'):
            raise ValueError('invalid genesis initialization')
        description=RANDOM_MANIFEST_DESCRIPTION if initialization=='random' else ''

        code_path = payload.get('code_path') or ''
        code_url = payload.get('code_url') or ''
        if code_path:
            file = Path(code_path).resolve()
            # Only code already vetted by the explicit upload endpoint may be registered.
            allowed = (self.workdir / '.fedpara-gui' / 'codes').resolve()
            if file.parent != allowed or not file.is_file():
                raise PermissionError('model code must first be uploaded through this dashboard')
        else:
            file = None
        manifest = make_hf_manifest(model,revision,source,architecture=task,
                                    seed=payload.get('seed',42),description=description,
                                    code_path=file,code_url=code_url)
        room_initialization(manifest)  # fail on non-pinned random models before writing
        # Preserve an imported code fingerprint if the user has not uploaded new code.
        with self.lock:
            old = self.imported.get(room)
            if old and not file and old['code']['sha256'] and old['code']['source']==code_url:
                manifest['code'] = dict(old['code'])
                validate_manifest(manifest)
            path = self._room_path(room)
            path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            os.chmod(path.parent,0o700)
            # Two WebViews share this pathname. An unchecked Save/Start could
            # replace a running peer's pinned genesis descriptor and induce
            # endless JOIN WAIT / genesis mismatches in another window.
            with _exclusive_manifest_lock(path.parent / '.manifest.lock'):
                if path.is_symlink():
                    raise PermissionError('symlink cannot be used as a room manifest')
                if path.is_file():
                    current=load_manifest(path)
                    if manifest_digest(current) != manifest_digest(manifest):
                        raise ValueError('このルームは既に別のモデル/データセット設定で固定されています。'
                                         '参加する場合は「設定を取得」で元の設定を読み込み、'
                                         '新しい構成に変える場合は「新規ルーム」を作成してください。')
                    return {'path':str(path),'sha256':manifest_digest(current),
                            'manifest':current}
                # Crash-safe commit, with exclusive cross-process descriptor lock.
                tmp=path.with_name('.manifest-'+secrets.token_hex(8))
                fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                try:
                    with os.fdopen(fd,'w',encoding='utf8') as out:
                        json.dump(manifest,out,ensure_ascii=False,indent=2,allow_nan=False)
                        out.write('\n')
                        out.flush();os.fsync(out.fileno())
                    os.replace(tmp,path)
                finally:
                    tmp.unlink(missing_ok=True)
            return {'path':str(path),'sha256':manifest_digest(manifest),'manifest':manifest}

    def load_room(self, payload: dict):
        room = validate_room(payload['room'])
        ws = check_signaling_url(payload['ws'])
        token = payload.get('token','')
        if not isinstance(token,str) or len(token)>256:
            raise ValueError('room token too large')
        manifest=asyncio.run(fetch_room_manifest(ws,room,token,timeout=8))
        if manifest['version'] != 2:
            raise ValueError('room uses a legacy dataset descriptor (v1)')
        with self.lock:
            self.imported[room]=manifest
        return {'manifest':manifest,'sha256':manifest_digest(manifest), 'warning':
                'Verify the manifest fingerprint with the room creator independently.'}

    def register_room(self, payload: dict):
        # Explicit network action. Saving a manifest locally remains possible
        # without Mummy online; published metadata is immutable thereafter.
        room=validate_room(payload['room'])
        path=self._room_path(room)
        if not path.is_file():
            raise FileNotFoundError('Save the room settings locally first')
        manifest=load_manifest(path)
        ws=check_signaling_url(payload.get('ws',''))
        asyncio.run(register_room_manifest(ws,room,manifest,
            publish=payload.get('room_public',False) is True,
            token=os.environ.get('APP_TOKEN',''),timeout=8))
        return {'registered':True,'room':room,'public':payload.get('room_public',False) is True}

    def list_rooms(self, payload: dict):
        # Read-only, bound to the user's chosen signaling host. Run on the
        # pywebview bridge worker thread; no HTTP service runs in this app.
        url = check_signaling_url(payload.get('ws',''))
        return {'rooms': list_public_rooms(url, timeout=5)}

    def upload_code(self, payload: dict):
        from .code_registry import register_base64
        return register_base64(self.workdir/'.fedpara-gui'/'codes',payload.get('name'),payload.get('contents'))

    def start(self, payload: dict):
        room=validate_room(payload['room'])
        with self.lock:
            path=self._room_path(room)
            if not path.is_file():
                raise FileNotFoundError('Save room settings before starting training')
            manifest=load_manifest(path)
            from .live_initialization import room_initialization
            initialization=room_initialization(manifest)
            raw_base = payload.get('base')
            if raw_base is None or (isinstance(raw_base,str) and not raw_base.strip()):
                # Automatic per-room immutable genesis cache; no path entry needed.
                # Different path from legacy/pretrained bootstrap prevents an
                # old cache silently defeating the requested random initialization.
                genesis_name=('genesis.random.fstate' if initialization=='random'
                              else 'genesis.fstate')
                base=(self.workdir/'.fedpara-gui'/'rooms'/room/genesis_name).resolve()
                automatic=True
            elif isinstance(raw_base,str):
                base=Path(raw_base.strip()).expanduser().resolve()
                automatic=False
            else:
                raise ValueError('invalid initial checkpoint path')
            if base.is_dir():
                raise IsADirectoryError('初期状態にはフォルダでなく .fstate ファイルが必要です: '+str(base))
            if room not in self.profiles:
                self.profiles[room] = allocate_profile(path.parent)
            profile = self.profiles[room]
            journal = profile.journal.resolve()
            reinit_every=payload.get('reinit_every',20)
            if type(reinit_every) is not int or not 0 <= reinit_every <= 100000:
                raise ValueError('reinit_every must be 0..100000 local commits')
            cmd=self.train.build_command(room=room,model=manifest['model']['id'],
                revision=manifest['model']['revision'],base=base,manifest=path,
                ws=check_signaling_url(payload['ws']),task=manifest['model']['architecture'],
                kind=payload.get('kind','loha'),rank=int(payload.get('rank',16)),journal=journal,
                optimizer=payload.get('optimizer','adamw'),reinit_every=reinit_every,
                initialization=initialization,initial_seed=manifest['datasets']['seed'],
                lr=payload.get('lr',1e-4),local_steps=payload.get('local_steps',64),
                max_length=payload.get('max_length',64),
                auto_state=automatic,room_public=payload.get('room_public',False) is True)
            pid=self.train.start(cmd,str(journal)+'.gui.log')
            return {'pid':pid,'journal':str(journal),'profile':profile.name,
                    'auto_state':automatic,'initialization':initialization,
                    'base':str(base)}

    def status(self):
        with self.lock:
            running=self.train.status()
            proc=self.train.proc
            return {'running':running,'pid':proc.pid if running else None,
                    'exit_code':proc.returncode if proc is not None and not running else None,
                    'logs':self.train.tail()}

    def copy_logs(self):
        """Copy the recent diagnostic log using a native clipboard command.

        Avoid navigator.clipboard, which is not consistently available to a
        local pywebview page on macOS. No shell interpolation is performed.
        """
        with self.lock:
            content=self.train.tail(max_bytes=2*1024*1024)
        if not content:
            raise ValueError('コピーできる学習ログがありません。')
        if sys.platform=='darwin':
            command=['pbcopy']
        elif sys.platform=='win32':
            command=['clip']
        else:
            command=next((cmd for cmd in (['wl-copy'],['xclip','-selection','clipboard'],['xsel','--clipboard','--input'])
                          if shutil.which(cmd[0])),None)
            if command is None:
                raise RuntimeError('クリップボードツールが見つかりません。ログをファイルへ保存してください。')
        try:
            subprocess.run(command, input=content, text=True, check=True,timeout=5,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            raise RuntimeError('クリップボードへのコピーに失敗しました。ログをファイルへ保存してください。') from exc
        return {'chars':len(content)}

    def save_logs(self):
        """Save the full local log to a private diagnostic file with a safe name."""
        with self.lock:
            logfile=self.train.logfile
            if logfile is None or not logfile.is_file():
                raise FileNotFoundError('保存する学習ログがまだありません。')
            output=self.workdir/'.fedpara-gui'/'diagnostics'
            output.mkdir(parents=True,exist_ok=True,mode=0o700)
            os.chmod(output,0o700)
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            dest=output/f'fedpara-{stamp}-{secrets.token_hex(4)}.log'
            fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            try:
                with os.fdopen(fd,'wb') as out,logfile.open('rb') as source:
                    shutil.copyfileobj(source,out,length=256*1024)
            except BaseException:
                dest.unlink(missing_ok=True)
                raise
            return {'path':str(dest),'bytes':dest.stat().st_size}

    def stop(self):
        with self.lock:
            self.train.stop()
            return {'stopping':True}

    def start_benchmark(self, payload: dict):
        """One local subprocess per GUI. This does not open signaling sockets."""
        with self.lock:
            if self._bench_proc is not None and self._bench_proc.poll() is None:
                raise RuntimeError('benchmark is already running in this GUI')
            if not isinstance(payload,dict):
                raise ValueError('expected benchmark options object')
            def option(name, default, lo, hi):
                value=payload.get(name,default)
                if type(value) is not int or not lo<=value<=hi:
                    raise ValueError(f'{name} must be an integer in {lo}..{hi}')
                return value
            peers=option('peers',4,2,16)
            rounds=option('rounds',5,1,200)
            steps=option('local_steps',2,1,40)
            reinit_every=option('reinit_every',20,0,200)
            model=payload.get('model','tinystories-1m')
            rank=option('rank',16 if model=='tinystories-1m' else 4,2,16)
            dataset=payload.get('dataset','tinystories' if model=='tinystories-1m' else 'auto')
            sequences=option('sequences',2048 if model=='tinystories-1m' else 64,1,8192)
            validation_sequences=option('validation_sequences',1024 if model=='tinystories-1m' else 64,16,8192)
            if model not in ('toy','tinystories-1m') or dataset not in ('auto','builtin','tinystories'):
                raise ValueError('invalid benchmark model or dataset')
            if model=='toy' and dataset!='auto':
                raise ValueError('toy model only supports automatic synthetic data')
            lr=payload.get('lr',1e-4 if model=='tinystories-1m' else 0.003)
            if type(lr) not in (float,int) or not 1e-6 <= lr <= .1:
                raise ValueError('invalid benchmark learning rate')
            initialization=payload.get('initialization','random')
            if initialization not in ('random','pretrained'):
                raise ValueError('invalid benchmark initialization')
            optimizer=payload.get('optimizer','adamw')
            if optimizer not in ('adamw','lora-muon'):
                raise ValueError('invalid benchmark optimizer')
            kind=payload.get('kind','loha')
            if kind not in ('loha','lora'):
                raise ValueError('invalid adapter kind')
            if optimizer=='lora-muon' and kind!='lora':
                raise ValueError('LoRA-Muon requires kind=lora')
            base=self.workdir/'.fedpara-gui'/'benchmarks'
            base.mkdir(parents=True,exist_ok=True,mode=0o700)
            os.chmod(base,0o700)
            dest=base/('run-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+secrets.token_hex(4))
            dest.mkdir(mode=0o700)
            logfile=dest/'benchmark.log'
            command=[sys.executable,'-m','patchlib.cli','benchmark-effectiveness',
                     '--peers',str(peers),'--rounds',str(rounds),'--local-steps',str(steps),
                     '--kind',kind,'--optimizer',optimizer,'--rank',str(rank),'--lr',str(lr),
                     '--reinit-every',str(reinit_every),
                     '--model',model,'--dataset',dataset,'--initialization',initialization,
                     '--sequences',str(sequences),'--validation-sequences',str(validation_sequences),
                     '--context','64' if model=='tinystories-1m' else '12',
                     '--output-dir',str(dest)]
            env=os.environ.copy()
            env['OMP_NUM_THREADS']='1'
            env['MKL_NUM_THREADS']='1'
            with logfile.open('wb') as out:
                self._bench_proc=subprocess.Popen(command,cwd=str(Path(__file__).resolve().parent.parent),
                     env=env,stdout=out,stderr=subprocess.STDOUT)
            self._bench_log=logfile
            self._bench_output=dest
            return {'pid':self._bench_proc.pid,'output_dir':str(dest)}

    def benchmark_status(self):
        with self.lock:
            proc=self._bench_proc
            if proc is None:
                return {'running':False,'exit_code':None,'logs':'','result':None}
            code=proc.poll()
            logfile=self._bench_log
            logs=''
            if logfile and logfile.is_file():
                with logfile.open('rb') as fh:
                    fh.seek(0,2);fh.seek(max(0,fh.tell()-5000))
                    logs=fh.read().decode('utf8','replace')
            summary=None
            report=self._bench_output/'effectiveness-result.json'
            if code==0 and report.is_file():
                # Report generated by our own fixed command, not a remote payload.
                content=json.loads(report.read_text(encoding='utf8'))
                summary={'loss':content['loss'],'compute':content['compute'],
                         'configuration':content['configuration'],'artifacts':{
                             k:str(self._bench_output/k) for k in
                             ('effectiveness-result.json','effectiveness-trace.csv',
                              'effectiveness-loss.png','effectiveness-time.png') if (self._bench_output/k).exists()},
                         'scope':content['scope']}
            return {'running':code is None,'exit_code':code,'logs':logs,
                    'result':summary,'output_dir':str(self._bench_output)}

    def close(self):
        # Do not release the identity while our training child is still alive.
        with self.lock:
            if self._bench_proc is not None and self._bench_proc.poll() is None:
                self._bench_proc.terminate()
                try:self._bench_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._bench_proc.kill();self._bench_proc.wait(timeout=3)
            self.train.stop(wait=True)
            for profile in self.profiles.values():
                profile.close()
            self.profiles.clear()

