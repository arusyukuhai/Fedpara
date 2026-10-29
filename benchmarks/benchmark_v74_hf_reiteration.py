"""Reproducible mocked HF dataset metadata open overhead (NOT real network)."""
import json
import time
from patchlib.hf_mixed_datasets_v68 import HFRowCycle

spec={'repo':'org/train','revision':'f'*40,'config':None,'split':'train','prompt_column':'prompt','response_column':'response'}
rows=[{'prompt':'a','response':'b'},{'prompt':'c','response':'d'}]

def run(reopen, n=200):
    calls=[0]
    def load(*args,**kwargs):
        calls[0]+=1
        time.sleep(.001)  # metadata setup costs 1ms, far below real Hub latency
        return rows
    obj=HFRowCycle(spec,'sft',dataset_factory=load)
    started=time.perf_counter()
    for i in range(n):
        if reopen and i and i%len(rows)==0:
            obj._open() # v73 equivalent metadata reload on each finite cycle
        obj.next()
    elapsed=time.perf_counter()-started
    obj.close()
    return {'seconds':round(elapsed,6),'load_dataset_calls':calls[0]}

if __name__=='__main__':
    print(json.dumps({'benchmark':'synthetic dataset construction delay 1ms, 200 rows (2 row split)',
                      'v73':run(True),'v74':run(False)},indent=2))
