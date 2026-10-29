"""Regression tests: evaluation ValueError sources on non-CUDA devices."""
from types import SimpleNamespace

import pytest

from patchlib.kagami_eval_worker import (_context_window, friendly_error,
                                        model_options, resolve_precision)
from patchlib.webview_controller import WebviewController
from patchlib.roomid import create_room_id
from patchlib.webview_gui import compose_html


class FakeTorch:
    class float32: pass
    class float16: pass
    class bfloat16: pass
    def __init__(self, *, cuda=False, bf16=False, mps=False):
        self.cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16)
        self.backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))


@pytest.mark.parametrize('cuda,bf16,mps,expected',[
    (False,False,False,'fp32'), (False,False,True,'fp16'),
    (True,False,False,'fp16'), (True,True,False,'bf16')])
def test_auto_precision_respects_actual_hardware(cuda,bf16,mps,expected):
    torch=FakeTorch(cuda=cuda,bf16=bf16,mps=mps)
    assert resolve_precision(torch,'auto')==expected
    options=model_options(torch,expected)
    assert options.get('dtype',options.get('torch_dtype')) is not None


def test_explicit_cpu_fp16_is_actionable_and_does_not_trap_auto():
    torch=FakeTorch()
    assert model_options(torch,resolve_precision(torch,'auto'))['device_map']=='cpu'
    with pytest.raises(ValueError,match='select AUTO'):
        model_options(torch,'fp16')


def test_context_budget_reserves_generation_and_handles_hf_sentinel():
    cfg=SimpleNamespace(n_positions=128, max_position_embeddings=None)
    model=SimpleNamespace(config=cfg)
    tokenizer=SimpleNamespace(model_max_length=10**30)
    assert _context_window(model,tokenizer,32)==(96,32)
    assert _context_window(model,tokenizer,128)==(64,64)
    cfg.n_positions=None
    assert _context_window(model,tokenizer,32)==(2016,32)


def test_diagnostic_hints_are_actionable_without_suppressing_original():
    assert 'Tokenizer' in friendly_error(ValueError('tokenizer vocab mismatch'))
    assert 'スナップショット' in friendly_error(ValueError('checkpoint identity mismatch'))
    assert 'FP32' in friendly_error(ValueError('CUDA quantization config'))


def test_gui_defaults_to_auto_and_supports_error_details():
    html=compose_html()
    assert '<option value="auto" selected>' in html
    assert 'state.result.hint' in html


def test_eval_accepts_auto_without_snapshot_when_base_explicit(tmp_path,monkeypatch):
    room=create_room_id()
    c=WebviewController(tmp_path)
    c.save({'room':room,'model':'demo/tiny','revision':'main','task':'causal-lm',
            'seed':42,'sources':[{'repo':'demo/text','revision':'main','config':None,
                                 'split':'train','columns':{'text':'text'},'weight':1.0}]})
    fake=SimpleNamespace(pid=5,poll=lambda:None,terminate=lambda:None)
    monkeypatch.setattr('patchlib.webview_controller.subprocess.Popen',lambda *a,**kw:fake)
    response=c.evaluate({'room':room,'scope':'base','prompt':'test','max_tokens':4})
    assert response['precision']=='auto'


def test_real_worker_allows_empty_prompt_and_adapts_tiny_context(tmp_path,monkeypatch):
    """Generation truly executes, not merely validating controller arguments."""
    import sys
    import types
    import torch
    from torch import nn
    from patchlib.kagami_eval_worker import run
    from patchlib.room_manifest import make_hf_manifest

    class ToyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.Embedding(32,8)
            self.config=SimpleNamespace(n_positions=16)
        def get_input_embeddings(self):return self.emb
        @classmethod
        def from_pretrained(cls,*args,**kwargs):return cls()
        def generate(self,input_ids,attention_mask=None,max_new_tokens=1,**kwargs):
            assert input_ids.tolist()==[[1]]
            assert max_new_tokens==8
            return torch.cat([input_ids,torch.tensor([[4]],device=input_ids.device)],dim=-1)

    class Tok:
        bos_token_id=1;eos_token_id=2;pad_token_id=0;model_max_length=10**30
        @classmethod
        def from_pretrained(cls,*args,**kwargs):return cls()
        def __call__(self, text,**kwargs):
            assert text==''
            return {'input_ids':torch.empty((1,0),dtype=torch.long)}
        def decode(self,ids,**kwargs):return 'generated'

    transformers=types.ModuleType('transformers')
    transformers.AutoTokenizer=Tok
    transformers.AutoModelForCausalLM=ToyLM
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    sources=[{'repo':'demo/text','revision':'main','config':None,'split':'train',
              'columns':{'text':'text'},'weight':1.0}]
    manifest=make_hf_manifest('demo/tiny','main',sources)
    result=run({'manifest':manifest,'prompt':'','precision':'auto','scope':'base',
                'max_tokens':64})
    assert result['text']=='generated'
    assert result['precision']=='fp32'
    assert result['requested_max_tokens']==64 and result['effective_max_tokens']==8


@pytest.mark.parametrize('task', ['seq2seq-lm','sequence-classification','masked-lm'])
def test_other_supported_model_tasks_run_real_forward(monkeypatch,task):
    import sys,types
    import torch
    from torch import nn
    from patchlib.kagami_eval_worker import run
    from patchlib.room_manifest import make_hf_manifest

    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(128,4)
            self.config=SimpleNamespace(n_positions=256,id2label={0:'NEG',1:'POS'})
        @classmethod
        def from_pretrained(cls,*a,**kw):return cls()
        def get_input_embeddings(self):return self.embedding
        def generate(self,**kw):return torch.tensor([[10,11,12]])
        def forward(self,input_ids,**kw):
            if task=='masked-lm':
                result=torch.zeros(1,input_ids.shape[1],128)
                result[:,0,15]=8
            else:
                result=torch.tensor([[0.,4.]])
            return SimpleNamespace(logits=result)

    class Tok:
        bos_token_id=1; eos_token_id=2;pad_token_id=0;mask_token_id=99;model_max_length=256
        @classmethod
        def from_pretrained(cls,*a,**kw):return cls()
        def __call__(self,text,**kw):return {'input_ids':torch.tensor([[99,20]])}
        def decode(self,ids,**kw):return ','.join(str(int(x)) for x in ids)

    transformers=types.ModuleType('transformers')
    transformers.AutoTokenizer=Tok
    transformers.AutoModelForSeq2SeqLM=Model
    transformers.AutoModelForSequenceClassification=Model
    transformers.AutoModelForMaskedLM=Model
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    sources=[{'repo':'demo/text','revision':'main','config':None,'split':'train',
              'columns':{'text':'text'},'weight':1.0}]
    manifest=make_hf_manifest('demo/model','main',sources,architecture=task)
    result=run({'manifest':manifest,'prompt':'[MASK]', 'precision':'auto',
                'scope':'base','max_tokens':12})
    assert result['task']==task
    if task=='seq2seq-lm':assert result['text']=='10,11,12'  # never trim encoder prompt
    if task=='sequence-classification':assert result['text'].startswith('POS (')
    if task=='masked-lm':assert result['text']=='15,20'
