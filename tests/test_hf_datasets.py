import pytest
from patchlib.core import PatchError
from patchlib.hf_datasets import rows_from_room
from patchlib.room_manifest import make_hf_manifest,manifest_digest,validate_manifest,load_manifest


def src(repo,column='text',weight=1,revision='main'):
    return {'repo':repo,'revision':revision,'config':None,'split':'train',
            'columns':{'text':column},'weight':weight}


class Stream:
    def __init__(self, rows):self.rows=rows
    def shuffle(self,**kwargs):return self
    def __iter__(self):return iter(self.rows)


def test_v2_collection_hash_and_validation(tmp_path):
    m=make_hf_manifest('org/model','main',[src('org/a'),src('org/b','body',3)],seed=42)
    assert m['version']==2
    assert len(manifest_digest(m))==64
    assert load_manifest(_dump(tmp_path,m))==m
    assert manifest_digest(m)!=manifest_digest(make_hf_manifest('org/model','main',[src('org/a')]))
    with pytest.raises(PatchError):make_hf_manifest('org/model','main',[src('http://evil/a')])
    with pytest.raises(PatchError):make_hf_manifest('org/model','main',[src('org/a',revision='../main')])
    with pytest.raises(PatchError):make_hf_manifest('org/model','main',[{**src('org/a'),'columns':{'exec':'python'}}])
    with pytest.raises(PatchError):validate_manifest({**m,'datasets':{'seed':42,'sources':[]}})


def _dump(tmp_path,m):
    import json
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m));return p


def test_streaming_weighted_rows_and_replay():
    m=make_hf_manifest('org/model','main',[src('org/a'),src('org/b','body',3)])
    calls=[]
    def loader(**kwargs):
        calls.append(kwargs)
        return Stream([{'text':'A'}] if kwargs['path']=='org/a' else [{'body':'B'}])
    one=rows_from_room(m,peer_id='alice',loader=loader)
    x=[next(one)['text'] for _ in range(100)]
    assert 50<x.count('B')<95
    assert len(calls)==2
    assert all(c['streaming'] is True for c in calls)
    assert [next(y)['text'] for y in [rows_from_room(m,peer_id='alice',loader=loader)]]==[x[0]]


def test_empty_dataset_fails():
    m=make_hf_manifest('org/model','main',[src('org/a')])
    with pytest.raises(PatchError,match='empty'):
        next(rows_from_room(m,loader=lambda **kw: Stream([])))


def test_dataset_values_not_executable():
    m=make_hf_manifest('org/model','main',[src('org/a')])
    row=next(rows_from_room(m,loader=lambda **kw: Stream([{'text':'print(42)'}])))
    assert row['text']=='print(42)'
