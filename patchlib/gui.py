"""Native Python desktop UI for Fedpara (Tkinter, no HTTP listener).

Dataset references are registered with the room, never proxied via Mummy.
All actions invoke fixed Python module commands without a shell.
"""
from __future__ import annotations
import json
import os
import subprocess
import threading
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .room_manifest import make_hf_manifest, manifest_digest
from .roomid import create_room_id, validate_room

ROOT = Path(__file__).resolve().parent.parent


class DesktopController:
    """Testable native app backend. No sockets, web browser or background service."""
    def __init__(self):
        self.proc = None
        self.logfile = None
        self.log_handle = None

    @staticmethod
    def create_manifest(model: str, revision: str, task: str, sources: list,
                        *, seed: int=42, code_path=None, code_url='') -> dict:
        return make_hf_manifest(model, revision, sources, architecture=task, seed=seed,
                                code_path=Path(code_path) if code_path else None,code_url=code_url)

    @staticmethod
    def save_manifest(path, manifest):
        from .room_manifest import validate_manifest
        validate_manifest(manifest)
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        return path

    @staticmethod
    def build_command(*,room,model,revision,base,manifest,ws,task,kind,rank,journal):
        from .room_manifest import load_manifest
        validate_room(room)
        desc=load_manifest(Path(manifest))
        if desc['version'] != 2 or desc['model']['id'] != model or desc['model']['revision'] != revision:
            raise ValueError('selected manifest and local model/revision disagree')
        if not Path(base).is_file():
            raise FileNotFoundError('initial .fstate checkpoint not found: '+str(base))
        if not ws.startswith(('ws://','wss://')) or len(ws)>512:
            raise ValueError('invalid signaling address')
        p=urlsplit(ws)
        if ws.startswith('ws://') and p.hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('remote signaling must use wss://')
        if task not in ('causal-lm','masked-lm','seq2seq-lm','sequence-classification'):
            raise ValueError('HF dataset UI currently supports text tasks only')
        if desc['model']['architecture'] != task:
            raise ValueError('manifest architecture differs from task')
        if kind not in ('lora','loha') or rank not in (4,8,16,32,64):
            raise ValueError('invalid adapter settings')
        if not str(journal).endswith('.sqlite3'):
            raise ValueError('journal must be a .sqlite3 path')
        return [sys.executable,'-m','patchlib.cli','train-live',
                '--room-id',room,'--model',model,'--revision',revision,
                '--hf-task',task,'--state-base',str(base),
                '--room-manifest',str(manifest),'--ws',ws,
                '--kind',kind,'--rank',str(rank),
                '--journal',str(journal),'--rounds','0']

    def start(self, command, logfile):
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError('training is already running')
        self.logfile=Path(logfile).resolve()
        self.logfile.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(self.logfile,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        self.log_handle=os.fdopen(fd,'ab',buffering=0)
        try:
            self.proc=subprocess.Popen(command,cwd=ROOT,
                       stdout=self.log_handle,stderr=subprocess.STDOUT)
        except BaseException:
            self.log_handle.close()
            raise
        return self.proc.pid

    def status(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.status():
            self.proc.terminate()
        if self.log_handle:
            self.log_handle.flush()

    def tail(self, max_bytes=10000):
        if self.logfile is None or not self.logfile.exists():
            return ''
        with self.logfile.open('rb') as f:
            f.seek(0,2)
            f.seek(max(0,f.tell()-max_bytes))
            return f.read(max_bytes).decode('utf8','replace')


class FederatedDesktop:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk=tk
        self.ttk=ttk
        self.root=root
        self.ctrl=DesktopController()
        self.imported_manifest=None
        root.title('Fedpara — P2P Training')
        root.geometry('900x820')
        frame=ttk.Frame(root,padding=14)
        frame.pack(fill='both',expand=True)
        self.v={name:tk.StringVar(value=default) for name,default in {
            'room':create_room_id(),'model':'sshleifer/tiny-gpt2',
            'revision':'main','task':'causal-lm',
            'base':'genesis.fstate','ws':'ws://127.0.0.1:8080/ws',
            'kind':'loha','rank':'32','journal':'gui-events.sqlite3',
            'manifest':'room-manifest.json','repo':'',
            'config':'','split':'train','dataset_revision':'main',
            'columns':'text=text','weight':'1.0','code':'',
            'code_url':'','seed':'42'}.items()}
        for idx,(name,label) in enumerate([
            ('room','Room ID'),('model','HF model ID'),('revision','Model revision'),
            ('task','Training task'),('base','Initial model state (.fstate)'),
            ('ws','Signaling WebSocket'),('manifest','Room manifest output'),
            ('journal','Local SQLite journal')]):
            ttk.Label(frame,text=label).grid(row=idx,column=0,sticky='w',padx=3,pady=3)
            if name=='task':
                w=ttk.Combobox(frame,textvariable=self.v[name],state='readonly',
                    values=('causal-lm','masked-lm','seq2seq-lm','sequence-classification'))
            else:
                w=ttk.Entry(frame,textvariable=self.v[name])
            w.grid(row=idx,column=1,columnspan=3,sticky='ew',padx=4,pady=3)
            if name=='room':
                ttk.Button(frame,text='New',command=lambda:self.v['room'].set(create_room_id())).grid(row=idx,column=4)
            if name in ('base','manifest'):
                ttk.Button(frame,text='Browse',command=lambda n=name:self.browse(n)).grid(row=idx,column=4)
        frame.grid_columnconfigure(1,weight=2)
        frame.grid_columnconfigure(2,weight=1)
        frame.grid_columnconfigure(3,weight=2)
        row=8
        ttk.Separator(frame,orient='horizontal').grid(row=row,column=0,columnspan=5,sticky='ew',pady=7)
        row+=1
        ttk.Label(frame,text='Hugging Face datasets (stored in the room, streamed at training time)',
                  font=('TkDefaultFont',11,'bold')).grid(row=row,column=0,columnspan=5,sticky='w',pady=5)
        row+=1
        self.tree=ttk.Treeview(frame,columns=('repo','config','split','revision','columns','weight'),show='headings',height=5)
        for col,width in [('repo',190),('config',100),('split',90),('revision',100),('columns',135),('weight',70)]:
            self.tree.heading(col,text=col)
            self.tree.column(col,width=width,stretch=True)
        self.tree.grid(row=row,column=0,columnspan=5,sticky='nsew')
        row+=1
        columns=[('repo','HF repo (org/name)'),('config','Config'),('split','Split'),
                 ('dataset_revision','Revision'),('columns','Columns'),('weight','Weight')]
        for col,(name,title) in enumerate(columns):
            # Draw second dataset editor as compact two-row grid.
            ttk.Label(frame,text=title).grid(row=row,column=col if col<5 else 4,sticky='w') if col<5 else None
        row+=1
        editors=ttk.Frame(frame)
        editors.grid(row=row,column=0,columnspan=5,sticky='ew')
        for col,(name,title) in enumerate(columns):
            ttk.Label(editors,text=title).grid(row=0,column=col,sticky='w')
            ttk.Entry(editors,textvariable=self.v[name],width=(24 if name=='repo' else 15)).grid(row=1,column=col,padx=2,sticky='ew')
        row+=1
        btn=ttk.Frame(frame)
        btn.grid(row=row,column=0,columnspan=5,sticky='w',pady=5)
        ttk.Button(btn,text='Add dataset',command=self.add_dataset).pack(side='left',padx=2)
        ttk.Button(btn,text='Remove selected',command=self.remove_dataset).pack(side='left',padx=2)
        ttk.Button(btn,text='Import collection JSON',command=self.import_sources).pack(side='left',padx=2)
        row+=1
        ttk.Separator(frame,orient='horizontal').grid(row=row,column=0,columnspan=5,sticky='ew',pady=8)
        row+=1
        ttk.Label(frame,text='Optional custom model code (.py or .zip; hash-only registration)').grid(row=row,column=0,columnspan=3,sticky='w')
        ttk.Entry(frame,textvariable=self.v['code']).grid(row=row+1,column=0,columnspan=4,sticky='ew')
        ttk.Button(frame,text='Browse',command=lambda:self.browse('code')).grid(row=row+1,column=4)
        row+=2
        ttk.Label(frame,text='Code HTTPS URL (optional)').grid(row=row,column=0,sticky='w')
        ttk.Entry(frame,textvariable=self.v['code_url']).grid(row=row,column=1,columnspan=4,sticky='ew')
        row+=1
        ttk.Label(frame,text='Adapter').grid(row=row,column=0,sticky='w')
        ttk.Combobox(frame,textvariable=self.v['kind'],values=('loha','lora'),state='readonly',width=10).grid(row=row,column=1,sticky='w')
        ttk.Label(frame,text='Rank').grid(row=row,column=2)
        ttk.Combobox(frame,textvariable=self.v['rank'],values=('4','8','16','32','64'),state='readonly',width=10).grid(row=row,column=3,sticky='w')
        row+=1
        self.state=tk.StringVar(value='Idle — room source collection not yet saved')
        ttk.Label(frame,textvariable=self.state).grid(row=row,column=0,columnspan=5,sticky='w',pady=7)
        row+=1
        btn2=ttk.Frame(frame)
        btn2.grid(row=row,column=0,columnspan=5,sticky='w')
        ttk.Button(btn2,text='Save room',command=self.save_room).pack(side='left',padx=4)
        ttk.Button(btn2,text='Join / load room datasets',command=self.join_room).pack(side='left',padx=4)
        ttk.Button(btn2,text='Start training',command=self.start).pack(side='left',padx=4)
        ttk.Button(btn2,text='Stop',command=self.stop).pack(side='left',padx=4)
        row+=1
        self.console=tk.Text(frame,height=12,wrap='word',state='disabled')
        self.console.grid(row=row,column=0,columnspan=5,sticky='nsew',pady=6)
        frame.grid_rowconfigure(row,weight=1)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(1000,self.poll)

    def _error(self,exc):
        from tkinter import messagebox
        messagebox.showerror('Fedpara',str(exc))

    def browse(self,key):
        from tkinter import filedialog
        val=(filedialog.askopenfilename() if key!='manifest' else
             filedialog.asksaveasfilename(defaultextension='.json'))
        if val:self.v[key].set(val)

    def add_dataset(self):
        try:
            cols=dict(part.strip().split('=',1) for part in self.v['columns'].get().split(',') if part.strip())
            entry={'repo':self.v['repo'].get().strip(),
                   'revision':self.v['dataset_revision'].get().strip(),
                   'config':self.v['config'].get().strip() or None,
                   'split':self.v['split'].get().strip(),
                   'columns':cols,'weight':float(self.v['weight'].get())}
            from .room_manifest import validate_hf_sources
            validate_hf_sources([entry])
            self.tree.insert('', 'end',values=(entry['repo'],entry['config'] or '',entry['split'],
                                               entry['revision'],','.join(f'{k}={v}' for k,v in cols.items()),entry['weight']))
        except Exception as exc:self._error(exc)

    def remove_dataset(self):
        for key in self.tree.selection():self.tree.delete(key)

    def import_sources(self):
        from tkinter import filedialog
        path=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not path:return
        try:
            from .room_manifest import validate_hf_sources
            sources=validate_hf_sources(json.loads(Path(path).read_text(encoding='utf8')))
            for entry in sources:
                self.tree.insert('','end',values=(entry['repo'],entry['config'] or '',entry['split'],entry['revision'],
                    ','.join(f'{k}={v}' for k,v in entry['columns'].items()),entry['weight']))
        except Exception as exc:self._error(exc)

    def sources(self):
        result=[]
        for ident in self.tree.get_children():
            repo,config,split,revision,columns,weight=self.tree.item(ident,'values')
            result.append({'repo':repo,'revision':revision,'config':config or None,'split':split,
                           'columns':dict(p.split('=',1) for p in columns.split(',')), 'weight':float(weight)})
        return result

    def save_room(self):
        try:
            m=self.ctrl.create_manifest(self.v['model'].get().strip(),self.v['revision'].get().strip(),
                self.v['task'].get(),self.sources(),seed=int(self.v['seed'].get()),
                code_path=self.v['code'].get().strip() or None,code_url=self.v['code_url'].get().strip())
            if self.imported_manifest is not None:
                from .room_manifest import manifest_digest
                # Preserve registered code SHA when no local code bundle exists.
                same_data=(m['model']==self.imported_manifest['model'] and
                           m['datasets']==self.imported_manifest['datasets'] and
                           m['code']['source']==self.imported_manifest['code']['source'])
                if same_data and not self.v['code'].get().strip():
                    m=self.imported_manifest
            path=self.ctrl.save_manifest(self.v['manifest'].get(),m)
            self.state.set('Saved '+str(path)+' | SHA-256 '+manifest_digest(m)[:16]+'…')
            return path
        except Exception as exc:self._error(exc);return None

    def join_room(self):
        import asyncio
        from tkinter import messagebox
        room=self.v['room'].get().strip()
        ws=self.v['ws'].get().strip()
        self.state.set('Fetching room metadata (no server-side dataset traffic)…')
        def worker():
            try:
                from .roomid import fetch_room_manifest
                manifest=asyncio.run(fetch_room_manifest(ws,room))
                if manifest['version']!=2:
                    raise ValueError('Room has legacy v1 metadata, not HF dataset collection')
                self.root.after(0,lambda:self.show_room(manifest))
            except Exception as exc:
                message=str(exc)
                self.root.after(0,lambda:self._error(message))
        threading.Thread(target=worker,daemon=True).start()

    def show_room(self, manifest):
        self.v['model'].set(manifest['model']['id'])
        self.v['revision'].set(manifest['model']['revision'])
        self.v['task'].set(manifest['model']['architecture'])
        self.v['seed'].set(str(manifest['datasets']['seed']))
        self.v['code_url'].set(manifest['code']['source'])
        self.v['code'].set('') # never auto-fetch or run remote code
        for ident in self.tree.get_children():self.tree.delete(ident)
        for entry in manifest['datasets']['sources']:
            self.tree.insert('','end',values=(entry['repo'],entry['config'] or '',entry['split'],entry['revision'],
                ','.join(f'{k}={v}' for k,v in entry['columns'].items()),entry['weight']))
        # Save exact server descriptor, including code hash; never recompute it.
        self.imported_manifest=manifest
        path=self.ctrl.save_manifest(self.v['manifest'].get(),manifest)
        self.state.set('Loaded and pinned '+str(path)+'; verify creator out-of-band')

    def start(self):
        manifest=self.save_room()
        if manifest is None:return
        try:
            args=self.ctrl.build_command(room=self.v['room'].get(),model=self.v['model'].get(),
                revision=self.v['revision'].get(),base=self.v['base'].get(),manifest=manifest,
                ws=self.v['ws'].get(),task=self.v['task'].get(),kind=self.v['kind'].get(),
                rank=int(self.v['rank'].get()),journal=self.v['journal'].get())
            pid=self.ctrl.start(args,self.v['journal'].get()+'.gui.log')
            self.state.set(f'Training running (PID {pid})')
        except Exception as exc:self._error(exc)

    def stop(self):
        self.ctrl.stop()
        self.state.set('Stop requested')

    def poll(self):
        if self.ctrl.proc and not self.ctrl.status():
            self.state.set(f'Training stopped (exit {self.ctrl.proc.returncode})')
        logs=self.ctrl.tail()
        if logs:
            self.console.configure(state='normal')
            self.console.delete('1.0','end')
            self.console.insert('end',logs)
            self.console.see('end')
            self.console.configure(state='disabled')
        self.root.after(1500,self.poll)

    def close(self):
        self.ctrl.stop()
        self.root.destroy()


def run_desktop():
    try:
        import tkinter as tk
        root=tk.Tk()
    except Exception as exc:
        raise RuntimeError('Tkinter GUI unavailable. Use desktop Python with Tk support and a display.') from exc
    FederatedDesktop(root)
    root.mainloop()
