"""Kagami v38: prevent accidental room forks without deleting histories."""
from pathlib import Path

import pytest

from patchlib.roomid import create_room_id
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import compose_html


def payload(room):
    return {
        'room': room, 'model': 'example/TinyModel', 'revision': 'main',
        'task': 'causal-lm', 'seed': 42, 'initialization': 'pretrained',
        'sources': [{'repo': 'example/Dataset', 'revision': 'main',
                     'config': None, 'split': 'train', 'columns': {'text': 'text'},
                     'weight': 1.0}],
        'prevent_duplicate': True,
    }


def test_same_model_as_new_id_requires_explicit_opt_in(tmp_path):
    ctl = WebviewController(tmp_path)
    room_a, room_b = create_room_id(), create_room_id()
    first = ctl.save(payload(room_a))
    assert Path(first['path']).is_file()
    with pytest.raises(ValueError, match='Identical model already saved under room') as exc:
        ctl.save(payload(room_b))
    assert room_a in str(exc.value)
    assert not ctl._room_path(room_b).exists()
    assert ctl._room_path(room_a).exists()
    assert len(ctl.local_models()['models']) == 1
    again = ctl.save(payload(room_a))
    assert again['sha256'] == first['sha256']


def test_independent_experiments_can_be_created_deliberately(tmp_path):
    ctl = WebviewController(tmp_path)
    room_a, room_b = create_room_id(), create_room_id()
    ctl.save(payload(room_a))
    independent = payload(room_b)
    independent['allow_duplicate'] = True
    second = ctl.save(independent)
    assert Path(second['path']).is_file()
    assert len(ctl.local_models()['models']) == 2
    assert ctl._room_path(room_a).exists()


def test_different_dataset_is_not_an_accidental_duplicate(tmp_path):
    ctl = WebviewController(tmp_path)
    room_a, room_b = create_room_id(), create_room_id()
    ctl.save(payload(room_a))
    different = payload(room_b)
    different['sources'][0]['split'] = 'validation'
    ctl.save(different)
    assert len(ctl.local_models()['models']) == 2


def test_training_button_never_creates_new_room():
    html = compose_html()
    assert 'let activatedRoom=null' in html
    assert "if(!activatedRoom||activatedRoom!==payload.room)" in html
    assert "else{await saveModel();}" not in html
    assert 'id="allowDuplicateModel"' in html
    assert 'data.prevent_duplicate=true' in html
    assert 'distinct IDs' in html


def test_corrupt_old_manifest_does_not_prevent_new_model(tmp_path):
    ctl = WebviewController(tmp_path)
    old_room = create_room_id()
    old_path = ctl._room_path(old_room)
    old_path.parent.mkdir(parents=True)
    old_path.write_text('{not-json', encoding='utf8')
    fresh = create_room_id()
    result = ctl.save(payload(fresh))
    assert Path(result['path']).is_file()
    assert old_path.read_text() == '{not-json'


def test_two_gui_windows_cannot_race_to_create_identical_model(tmp_path):
    import concurrent.futures
    room_a, room_b = create_room_id(), create_room_id()
    def attempt(room):
        c = WebviewController(tmp_path)
        try:
            c.save(payload(room))
            return 'created'
        except ValueError as error:
            assert 'Identical model already saved' in str(error)
            return 'duplicate-blocked'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (room_a, room_b)))
    assert sorted(results) == ['created', 'duplicate-blocked']
    assert len(WebviewController(tmp_path).local_models()['models']) == 1


def test_duplicate_report_is_read_only(tmp_path):
    from scripts.report_duplicate_model_rooms import duplicate_groups
    ctl = WebviewController(tmp_path)
    room_a, room_b = create_room_id(), create_room_id()
    ctl.save(payload(room_a))
    second = payload(room_b)
    second['allow_duplicate'] = True
    ctl.save(second)
    before = ctl._room_path(room_b).read_bytes()
    result = duplicate_groups(tmp_path)
    assert len(result) == 1
    assert {item['room_id'] for item in result[0]['rooms']} == {room_a, room_b}
    assert ctl._room_path(room_b).read_bytes() == before


def test_joined_remote_model_uses_same_network_id_and_is_imported(tmp_path, monkeypatch):
    from patchlib.room_manifest import make_hf_manifest, manifest_digest
    from patchlib.model_design import DESCRIPTION
    from patchlib.webview_controller import WebviewController
    room = create_room_id()
    descriptor = make_hf_manifest('example/TinyModel', 'main', payload(room)['sources'],
                                  description=DESCRIPTION['weights'])
    descriptor['adapter']={'kind':'loha','rank':16,'wide_lora_rank':4}
    async def fetch(url, rid, token, timeout):
        assert rid == room
        return descriptor
    monkeypatch.setattr('patchlib.webview_controller.fetch_room_manifest',fetch)
    ctl = WebviewController(tmp_path)
    read = ctl.select_model({'room':room,'ws':'ws://127.0.0.1:8080/ws'})
    assert read['local'] is False
    assert read['sha256'] == manifest_digest(descriptor)
    # UI imports the pinned remote descriptor before enabling TRAINING.
    imported = payload(room)
    imported['prevent_duplicate'] = False
    imported['adapter_kind']='loha'
    result = ctl.save(imported)
    assert result['sha256'] == read['sha256']
    assert ctl.local_models()['models'][0]['room_id'] == room
    assert ctl.select_model({'room':room})['local'] is True
