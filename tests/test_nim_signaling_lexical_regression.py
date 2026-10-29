"""Guard Nim source against operator-adjacency parse errors.

This static check does not replace `nim check` with a Nim 2 compiler.
"""
from pathlib import Path
import re


NIM = Path(__file__).resolve().parents[1] / 'src' / 'main.nim'


def test_assignment_operator_does_not_glue_to_json_operators():
    text = NIM.read_text(encoding='utf-8')
    # `notice=$(` and `reply[...] =%` are lexed as custom infix operators.
    assert not re.search(r'(?<![=!<>])=(?=[%$@+*/<>!&|^-])', text)


def test_group_neighborhood_notice_remains_json_encoded():
    text = NIM.read_text(encoding='utf-8')
    assert 'let notice = $(%*{' in text
    for key in ('topology', 'degree', 'online', 'revision'):
        assert f'reply["{key}"] = %' in text
