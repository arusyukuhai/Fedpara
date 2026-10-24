#!/usr/bin/env python3
"""Summarize repeated WebRTC reconnect causes from a local Swarm run.

Usage: python scripts/analyze_swarm_connection_failures.py swarm-runs/your-run
This reads only local run-*.log files and does not contact the network.
"""
import argparse
from collections import Counter
from pathlib import Path
import json
import re


def diagnose(run: Path):
    counter=Counter()
    samples={}
    per_peer={}
    errors=[]
    log_files=sorted(run.glob('peer-*/run-*.log'))
    if not log_files:
        raise FileNotFoundError(f'No peer-*/run-*.log files under {run}')
    markers={
        'link_failure':'WebRTC link failure',
        'signal_peer_left':'P2P signaling peer-left',
        'sctp_closed':'SCTP DataChannel closed',
        'datachannel_closed':'phase=datachannel-closed',
        'negotiation_timeout':'negotiation timeout',
        'heartbeat_stalled':'signed heartbeat stalled',
        'protocol_rejected':'peer frame rejected',
        'gossip_tx_failed':'signed gossip transmit failed',
        'queue_overflow':'critical control queue saturated',
        'websocket_ping_timeout':'keepalive ping timeout',
        'webrtc_sendto_none':"'NoneType' object has no attribute 'sendto'",
        'webrtc_loop_none':"'NoneType' object has no attribute 'call_exception_handler'",
        'invalid_state':'InvalidStateError',
    }
    for log in log_files:
        lines=log.read_text(errors='replace').splitlines()
        local=Counter()
        for i,line in enumerate(lines):
            for key,marker in markers.items():
                if marker in line:
                    local[key]+=1;counter[key]+=1
                    samples.setdefault(key,{'file':str(log),'line':i+1,
                        'excerpt':lines[max(0,i-4):min(len(lines),i+3)]})
                    if key in ('webrtc_sendto_none','webrtc_loop_none','invalid_state'):
                        if len(errors)<12:
                            errors.append({'file':str(log),'line':i+1,
                                'context':lines[max(0,i-10):min(len(lines),i+2)]})
        per_peer[str(log.relative_to(run))]=dict(local)
    return {'log_files':len(log_files),'event_counts':dict(counter),
            'per_log':per_peer,'samples':samples,'traceback_excerpts':errors}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    summary=diagnose(a.run)
    dest=a.output or a.run/'connection-diagnostic.json'
    dest.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(summary['event_counts'],indent=2))
    print(f'Wrote {dest}')


if __name__ == '__main__':
    main()
