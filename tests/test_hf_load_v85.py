"""v85 regressions: missing-model_type at tokenizer/model secondary loading."""
import json
import sys
from types import SimpleNamespace

import pytest

from patchlib.hf_config_v83 import MissingModelTypeError, load_hf_config
from patchlib.hf_load_v85 import (is_missing_model_type_error,
                                   load_hf_model, load_hf_tokenizer)

ERR='Unrecognized model. Should have a `model_type` key in its config.json'


def _legacy(tmp_path, architecture='GPTNeoForCausalLM'):
    (tmp_path/'config.json').write_text(json.dumps({'architectures':[architecture],
        'vocab_size':64, 'hidden_size':32}),encoding='utf8')
    return str(tmp_path)


@pytest.fixture
def fake_hf(monkeypatch):
    records=[]
    class AutoConfig:
        @staticmethod
        def from_pretrained(*args,**kw):
            # HF's path-name fallback could return an incorrect config.
            records.append(('AutoConfig.from_pretrained', args, kw))
            return SimpleNamespace(model_type='gpt2')
        @staticmethod
        def for_model(kind,**kw):
            records.append(('AutoConfig.for_model',kind,kw))
            return SimpleNamespace(model_type=kind, **kw)
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args,**kw):
            records.append(('AutoTokenizer.from_pretrained',args,kw))
            if 'config' not in kw:raise ValueError(ERR)
            return SimpleNamespace(model_type=kw['config'].model_type)
    class AutoModel:
        @staticmethod
        def from_pretrained(*args,**kw):
            records.append(('AutoModel.from_pretrained',args,kw))
            if 'config' not in kw:raise ValueError(ERR)
            return SimpleNamespace(model_type=kw['config'].model_type)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoConfig=AutoConfig,AutoTokenizer=AutoTokenizer))
    return records,AutoModel


def test_initial_config_avoids_incorrect_hf_name_fallback(tmp_path,fake_hf):
    records,_=fake_hf
    model=tmp_path/'my-gpt2-named-folder'; model.mkdir()
    load=_legacy(model)
    config=load_hf_config(load,local_files_only=True)
    assert config.model_type=='gpt_neo' # not path-name gpt2
    assert not any(event[0]=='AutoConfig.from_pretrained' for event in records)


def test_tokenizer_can_recover_but_never_edit_source(tmp_path,fake_hf):
    records,_=fake_hf
    model=_legacy(tmp_path)
    original=(tmp_path/'config.json').read_bytes()
    token=load_hf_tokenizer(model,revision='pinned',local_files_only=True)
    assert token.model_type=='gpt_neo'
    assert (tmp_path/'config.json').read_bytes()==original
    assert [x[0] for x in records].count('AutoTokenizer.from_pretrained')==2
    assert records[-1][-1]['revision']=='pinned'
    assert records[-1][-1]['trust_remote_code'] is False


def test_reward_or_critic_model_can_recover_with_config(tmp_path,fake_hf):
    records,cls=fake_hf
    model=_legacy(tmp_path)
    loaded=load_hf_model(cls,model,revision='pinned',local_files_only=True,
                         use_safetensors=True,output_loading_info=True)
    assert loaded.model_type=='gpt_neo'
    assert records[-1][-1]['use_safetensors'] is True
    assert records[-1][-1]['output_loading_info'] is True


def test_unknown_architecture_fails_instead_of_guessing(tmp_path,fake_hf):
    model=_legacy(tmp_path,'UnknownCustomTransformer')
    with pytest.raises(MissingModelTypeError,match='model_type'):
        load_hf_tokenizer(model,local_files_only=True)
    with pytest.raises(MissingModelTypeError,match='model_type'):
        load_hf_config(model,local_files_only=True)


def test_unrelated_value_error_not_swallowed(monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*args,**kw):raise ValueError('Tokenizer vocabulary invalid')
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=FakeTokenizer))
    with pytest.raises(ValueError,match='vocabulary invalid'):
        load_hf_tokenizer('sample/sample')
    assert is_missing_model_type_error(ValueError(ERR))
    assert not is_missing_model_type_error(ValueError('Unrecognized tokenizer'))


def test_provided_config_does_not_call_autoconfig(fake_hf):
    records,_=fake_hf
    token=load_hf_tokenizer('any/local',config=SimpleNamespace(model_type='llama'))
    assert token.model_type=='llama'
    assert [x[0] for x in records]==['AutoTokenizer.from_pretrained']
