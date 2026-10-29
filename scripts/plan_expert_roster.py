#!/usr/bin/env python3
"""Build an auditable, deterministic expert ownership map from a resource report."""
import argparse
import json
from pathlib import Path
from patchlib.moe_planner import ExpertGroup, plan_expert_placement


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True,
                        help='JSON produced by --resource-report')
    parser.add_argument('--roster',type=Path,required=True,
                        help='JSON {"peer_id":{"weight":N,"expert_budget_bytes":N}, ...}')
    parser.add_argument('--room',required=True)
    parser.add_argument('--generation',required=True)
    parser.add_argument('--replicas',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    report=json.loads(a.report.read_text())
    rows=report['expert_shapes']
    groups=[ExpertGroup(**r) for r in rows]
    roster=json.loads(a.roster.read_text())
    if not isinstance(roster,dict) or not roster:
        raise ValueError('empty roster')
    weights={peer:int(value['weight']) for peer,value in roster.items()}
    budgets={peer:int(value['expert_budget_bytes']) for peer,value in roster.items()}
    placement=plan_expert_placement(groups,a.room,weights,replicas=a.replicas,
                                    generation=a.generation,peer_byte_budgets=budgets)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(placement,indent=2),encoding='utf-8')
    print(f'Planned {len(placement["experts"])} experts in {a.output}')

if __name__=='__main__':main()
