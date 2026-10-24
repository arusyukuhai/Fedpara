"""CLI-facing preflight: fail clearly rather than silently running fake WebRTC."""
import subprocess
import sys
from pathlib import Path


def test_swarm_cli_lists_real_transport_options():
    proc=subprocess.run([sys.executable,'-m','patchlib.cli','swarm-sim','--help'],
                        capture_output=True,text=True,check=True,timeout=20)
    assert '--embedded-signaling' in proc.stdout
    assert '--faults' in proc.stdout
    assert '--settle-timeout' in proc.stdout


def test_swarm_missing_dependency_does_not_create_directory(tmp_path):
    import importlib.util
    if all(importlib.util.find_spec(x) for x in ('transformers','aiortc')):
        return  # Only exercise missing-deps branch when it is actually missing.
    target=tmp_path/'new-test'
    proc=subprocess.run([sys.executable,'-m','patchlib.cli','swarm-sim',
                         '--embedded-signaling','--output',str(target)],
                        capture_output=True,text=True,timeout=20)
    assert proc.returncode != 0
    assert not target.exists()
    assert 'requires real dependencies' in proc.stderr
