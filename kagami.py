"""Kagami desktop application launcher.

Usage: python kagami.py
All legacy CLI commands remain available through python -m patchlib.cli.
"""
from patchlib.webview_gui import run_webview

if __name__ == '__main__':
    run_webview()
