"""Reproducible PAGE COW allocation/CPU I/O microbench; not a training benchmark."""
import argparse
import json
import tempfile
import time
from pathlib import Path
import torch
from patchlib.paged_mutable_v61 import SparsePageStore


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='BENCHMARK_V61.json')
    args=parser.parse_args()
    torch.manual_seed(17)
    base=torch.randn((2048,2048),dtype=torch.float32) # 16 MiB tensor
    with tempfile.TemporaryDirectory() as d:
        t=time.perf_counter()
        store=SparsePageStore(base,Path(d)/'sparse',page_bytes=65536)
        init=time.perf_counter()-t
        t=time.perf_counter()
        values=store.rows(17,18);values[0,5]+=1
        store.write_rows(17,values)
        edit=time.perf_counter()-t
        st=(Path(d)/'sparse').stat()
        output={
            'warning':'Single local CPU sparse-edits storage microbench; NOT 300B, WAN, LoHA dense-update, or end-to-end training.',
            'tensor_bytes':base.numel()*base.element_size(),
            'page_bytes':65536,
            'changed_pages':store.stats()['changed_pages'],
            'logical_sparse_file_bytes':st.st_size,
            'allocated_blocks_bytes':st.st_blocks*512,
            'page_store_accounted_bytes':store.stats()['allocated_page_bytes'],
            'initialization_seconds':init,
            'one_row_edit_seconds':edit,
        }
        assert output['changed_pages']<=2
        assert torch.equal(base[17],base[17].clone())
        store.close()
        dense=SparsePageStore(base,Path(d)/'dense',page_bytes=65536)
        start=time.perf_counter()
        for r in range(0,base.shape[0],128):
            block=dense.rows(r,min(r+128,base.shape[0]))
            dense.write_rows(r,block+1)
        output['all_pages_modified_seconds']=time.perf_counter()-start
        output['all_pages_modified_accounted_bytes']=dense.stats()['allocated_page_bytes']
        output['all_pages_modified_disk_blocks_bytes']=(Path(d)/'dense').stat().st_blocks*512
        dense.close()
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':run()
