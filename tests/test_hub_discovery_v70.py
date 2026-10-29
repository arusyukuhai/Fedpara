"""v70: purpose-specific discovery is safe, bounded, and never claims compatibility."""
from types import SimpleNamespace as Row

import pytest

from patchlib.hub_discovery_v70 import discover_hub_assets
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import FedparaBridge, compose_html

SHA='a'*40

class FakeHub:
    def __init__(self): self.calls=[]
    def list_datasets(self, **opts):
        self.calls.append(('dataset', opts))
        return [
            Row(id='sample/educational-instruction-sft',sha=SHA,downloads=120,tags=['text','instruction'],gated=False),
            Row(id='sample/rlhf-prompts',sha=SHA,downloads=190,tags=['prompt','rlhf'],gated=False),
            Row(id='sample/fineweb-corpus',sha=SHA,downloads=10000,tags=['task_ids:language-modeling'],gated=False),
        ]
    def list_models(self, **opts):
        self.calls.append(('model', opts))
        return [
            Row(id='sample/reward-model',sha=SHA,downloads=1600,tags=['reward'],pipeline_tag='text-classification',gated=True),
            Row(id='sample/value-head-critic',sha=SHA,downloads=500,tags=['value-head','critic'],pipeline_tag='text-classification',gated=False),
            Row(id='sample/sentiment-classifier',sha=SHA,downloads=25000,tags=['sentiment'],pipeline_tag='text-classification',gated=False),
        ]

@pytest.mark.parametrize('role,expected',[
    ('pt','sample/fineweb-corpus'),('sft','sample/educational-instruction-sft'),
    ('ppo','sample/rlhf-prompts'),('reward','sample/reward-model'),
    ('critic','sample/value-head-critic')])
def test_default_discovery_ranks_purpose_without_text_query(role,expected):
    hub=FakeHub();res=discover_hub_assets({'role':role},api=hub)
    assert res['role']==role and res['scope']=='recommended'
    assert res['candidates'][0]['id']==expected
    assert all(c['requires_inspection'] and c['fit']=='candidate' for c in res['candidates'])
    assert len(hub.calls)<=3
    if role=='pt':assert all(call[1]['filter']=='task_ids:language-modeling' for call in hub.calls)
    if role in ('reward','critic'):assert all(call[1]['pipeline_tag']=='text-classification' for call in hub.calls)
    assert all(call[1]['limit']<=20 for call in hub.calls)


def test_broader_search_is_opt_in_and_does_not_claim_all_models_work():
    hub=FakeHub();result=discover_hub_assets({'role':'critic','scope':'broader','query':'sentiment'},api=hub)
    assert hub.calls==[('model',{'sort':'downloads','limit':20,'search':'sentiment'})]
    assert any(c['id']=='sample/sentiment-classifier' and c['requires_inspection'] for c in result['candidates'])

@pytest.mark.parametrize('obj',[
    {}, {'role':'wrong'}, {'role':'sft','scope':'hack'},
    {'role':'ppo','query':'x'*101}, 'ppo',
    {'role':'reward','query':[]},
])
def test_invalid_requests_fail_closed(obj):
    with pytest.raises(ValueError):discover_hub_assets(obj,api=FakeHub())


def test_bridge_and_gui_include_picker(tmp_path):
    c=WebviewController(tmp_path)
    from unittest.mock import patch
    with patch('patchlib.hub_discovery_v70.discover_hub_assets',return_value={'candidates':[]}) as find:
        assert FedparaBridge(c).discover_hub_assets({'role':'pt'})=={'candidates':[]}
        find.assert_called_once()
        assert find.call_args.args==({'role':'pt'},)
        assert find.call_args.kwargs['cache'] is c._hub_discovery_cache
    html=compose_html()
    for key in ['hubPicker','hubPickerScope','hubPickerQuery','hubPickerResults','ptHubBrowse','mixSftHfSearch','mixCriticSearch','mixRewardSearch','mixPpoHfSearch']:
        assert 'id="'+key+'"' in html
    assert 'hubDiscover' in html


def test_network_failure_is_not_silently_hidden():
    class Failure:
        def list_models(self,**kw): raise RuntimeError('network down')
    with pytest.raises(RuntimeError,match='unavailable'):
        discover_hub_assets({'role':'reward'},api=Failure())
