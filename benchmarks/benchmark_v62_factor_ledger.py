"""Synthetic dense low-rank update: factor memory vs page COW disk cost.

This is not an end-to-end training or 300B performance benchmark.
"""
import json, tempfile, time
from pathlib import Path
import torch
from patchlib.core import FactorDelta
from patchlib.factor_ledger_v62 import FactorLedgerSource
from patchlib.paged_mutable_v61 import SparsePageStore


def run(rows=2048,cols=2048,rank=8):
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(621)
    base=torch.zeros((rows,cols),dtype=torch.float32)
    spec=FactorDelta((rows,cols),'loha',rank,rank)
    now=[torch.randn_like(p)*.1 for p in spec.parts]
    old=[torch.zeros_like(p) for p in spec.parts]
    with tempfile.TemporaryDirectory() as tmp:
        sparse=SparsePageStore(base,Path(tmp)/'pages',page_bytes=65536)
        ledger=FactorLedgerSource(base,spec,max_entries=32)
        t0=time.perf_counter()
        ledger.append(now,old)
        factor_write_ms=(time.perf_counter()-t0)*1000
        # Prepare full dense update solely for the v61 control workload.
        from patchlib.livefed import _delta_block
        full=_delta_block(spec,now,0,rows)
        t0=time.perf_counter()
        for lo in range(0,rows,16):
            hi=min(rows,lo+16)
            sparse.write_rows(lo,full[lo:hi])
        page_write_ms=(time.perf_counter()-t0)*1000
        t0=time.perf_counter()
        ledg_tile=ledger[3:67,0:cols]
        factor_read_ms=(time.perf_counter()-t0)*1000
        t0=time.perf_counter()
        page_tile=sparse[3:67,0:cols]
        page_read_ms=(time.perf_counter()-t0)*1000
        initial_factor_bytes=ledger.stats()['factor_bytes']
        err=(ledg_tile-page_tile).abs().max().item()
        tile_scaling={'1':round(factor_read_ms,4)}
        # Worst-case retained update history; all entries are read on each tile.
        for count in (8,32):
            for _ in range(count-len(ledger.entries)):
                ledger.append(now,old)
            t0=time.perf_counter()
            _=ledger[3:67,0:cols]
            tile_scaling[str(count)]=round((time.perf_counter()-t0)*1000,4)
        report={
          'kind':'synthetic-cpu-single-loha-event','rows':rows,'cols':cols,'rank':rank,
          'base_bytes':base.numel()*base.element_size(),
          'factor_ledger_bytes':initial_factor_bytes,
          'page_cow_written_bytes':sparse.stats()['allocated_page_bytes'],
          'factor_append_ms':round(factor_write_ms,4),
          'page_cow_update_ms':round(page_write_ms,4),
          'factor_tile_read_ms':round(factor_read_ms,4),
          'page_tile_read_ms':round(page_read_ms,4),
          'max_abs_tile_diff':err,
          'factor_tile_read_scaling_ms':tile_scaling,
          'note':'Control path receives already-computed dense update; not end-to-end validated event time. Inference tile time rises with retained event count.'
        }
        sparse.close()
        return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
