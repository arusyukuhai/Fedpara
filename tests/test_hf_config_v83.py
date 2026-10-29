"""v83 regression: missing Hugging Face model_type must be actionable and safe."""
import json
import sys
import types
import pytest
from patchlib.hf_config_v83 import (checked_model_type, load_hf_config,
    save_hf_config_with_model_type, MissingModelTypeError)
from patchlib.hf_compat_v63 import diagnose_hf_checkpoint, validate_training_config


def test_explicit_model_type_takes_priority():
    assert checked_model_type({'model_type':'gpt_neo','architectures':['GPT2LMHeadModel']})=='gpt_neo'


@pytest.mark.parametrize('architecture,expected', [
    ('GPT2LMHeadModel','gpt2'),('GPTNeoForCausalLM','gpt_neo'),
    ('LlamaForCausalLM','llama'),('Qwen2ForCausalLM','qwen2'),
    ('MistralForCausalLM','mistral'),('Gemma2ForCausalLM','gemma2'),
])
def test_only_unambiguous_builtin_inference(architecture, expected):
    assert checked_model_type({'architectures':[architecture]})==expected


@pytest.mark.parametrize('data', [
    {}, {'model_type':''}, {'model_type':3}, {'architectures':['UntrustedModel']},
    {'architectures':['GPT2LMHeadModel','LlamaForCausalLM']},
    {'architectures':['GPT2LMHeadModel','EvilForCausalLM']},
])
def test_unsafe_missing_type_rejected(data):
    with pytest.raises(MissingModelTypeError, match='model_type'):
        checked_model_type(data)


def test_missing_type_with_builtin_architecture_recovers_without_mutating_source(tmp_path,monkeypatch):
    data={'architectures':['GPT2LMHeadModel'],'vocab_size':71,'n_embd':24}
    path=tmp_path/'config.json';path.write_text(json.dumps(data))
    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls,*a,**k):
            raise ValueError('Should have a model_type key in config.json')
        @classmethod
        def for_model(cls, model_type,**options):
            assert options['vocab_size']==71
            return types.SimpleNamespace(model_type=model_type, **options)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    result=load_hf_config(str(tmp_path), local_files_only=True)
    assert result.model_type=='gpt2'
    assert json.loads(path.read_text())==data # no silent mutation of a pinned source


def test_unknown_model_type_is_not_replaced_with_guess(tmp_path,monkeypatch):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'unknown99','architectures':['GPT2LMHeadModel']}))
    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls,*a,**k):raise ValueError('Unknown model type')
        @classmethod
        def for_model(cls,*a,**k):pytest.fail('must never substitute a different architecture')
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoConfig=FakeAutoConfig))
    with pytest.raises(ValueError,match='Unknown model type'):
        load_hf_config(str(tmp_path),local_files_only=True)


def test_unsafe_config_rejected_at_diagnosis(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['UnknownModel']}))
    (tmp_path/'model.safetensors').write_bytes(b'placeholder')
    report=diagnose_hf_checkpoint(str(tmp_path),local_files_only=True)
    assert not report['streaming_candidate']
    assert report['model_type'] is None
    assert 'model_type' in '\n'.join(report['notes'])
    assert not validate_training_config_safe(types.SimpleNamespace())


def validate_training_config_safe(config):
    try:
        validate_training_config(config,'foo/bar')
    except ValueError:
        return False
    return True


def test_diagnosis_reports_safe_inference(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['LlamaForCausalLM']}))
    report=diagnose_hf_checkpoint(str(tmp_path),local_files_only=True)
    assert report['model_type']=='llama'
    assert report['streaming_candidate']
    assert any('inferred' in x for x in report['notes'])


class IncompleteSerializer:
    model_type='gpt2'
    def save_pretrained(self, target):
        (target/'config.json').write_text('{"architectures":["GPT2LMHeadModel"],"vocab_size":32}')


def test_saved_config_contains_model_type_even_if_serializer_omits_it(tmp_path):
    assert save_hf_config_with_model_type(IncompleteSerializer(),tmp_path)=='gpt2'
    assert json.loads((tmp_path/'config.json').read_text())['model_type']=='gpt2'


def test_export_does_not_overwrite_conflicting_explicit_type(tmp_path):
    class IncorrectSerializer(IncompleteSerializer):
        def save_pretrained(self,target):
            (target/'config.json').write_text('{"model_type":"llama"}')
    with pytest.raises(MissingModelTypeError, match='conflicts'):
        save_hf_config_with_model_type(IncorrectSerializer(),tmp_path)


def test_repair_tool_explicit_backup_and_identity(tmp_path):
    from scripts.fix_hf_model_type_v83 import repair_local
    source={'architectures':['GPTNeoForCausalLM'],'vocab_size':101}
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(source))
    assert repair_local(tmp_path)=='gpt_neo'
    assert json.loads(cfg.read_text())==source
    assert repair_local(tmp_path,write=True)=='gpt_neo'
    assert json.loads(cfg.read_text())['model_type']=='gpt_neo'
    assert json.loads((tmp_path/'config.json.before-v83.bak').read_text())==source
    with pytest.raises(ValueError,match='different'):
        repair_local(tmp_path,model_type='gpt2',write=True)


def test_repair_tool_never_guesses_unknown_model(tmp_path):
    from scripts.fix_hf_model_type_v83 import repair_local
    config=tmp_path/'config.json';config.write_text('{"architectures":["MyCustomModel"]}')
    with pytest.raises(MissingModelTypeError):
        repair_local(tmp_path,write=True)
    assert not (tmp_path/'config.json.before-v83.bak').exists()
    assert repair_local(tmp_path,model_type='llama',write=False)=='llama'


def test_merge_export_enforces_config_model_type(tmp_path):
    from patchlib.cli import save_model
    class Model:
        config = IncompleteSerializer()
        def save_pretrained(self, folder, safe_serialization=True):
            (tmp_path/'model.safetensors').write_bytes(b'dummy')
    save_model(Model(),tmp_path)
    assert json.loads((tmp_path/'config.json').read_text())['model_type']=='gpt2'
