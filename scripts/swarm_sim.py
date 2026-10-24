#!/usr/bin/env python3
"""Convenience wrapper for python -m patchlib.cli swarm-sim."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patchlib.cli import main
if __name__=='__main__':
    sys.argv.insert(1,'swarm-sim')
    main()
