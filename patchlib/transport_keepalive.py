"""Signaling WebSocket keepalive policy for busy multi-process trainers.

When two macOS GUI windows run models and stream datasets simultaneously,
Python event loops can suffer occasional long scheduling delays. An aiortc
DataChannel has its own SCTP/ICE health; a delayed *local* WS Pong is not a
reason to tear it down. Loopback TCP reports actual process exit to the OS,
while lightweight application Ping messages keep the signaling session active.
Internet WSS still uses a generous bounded native WebSocket ping timeout.
"""
from urllib.parse import urlsplit


def signaling_keepalive(url: str) -> dict:
    host = urlsplit(url).hostname
    if host in ('localhost', '127.0.0.1', '::1'):
        return {'ping_interval': None, 'ping_timeout': None}
    return {'ping_interval': 30, 'ping_timeout': 150}
