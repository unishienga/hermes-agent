"""A `calls` string that will not parse must be recoverable, not a dead end.

Control (pre-change behaviour, reproduced live): the rejection was the bare
``json.JSONDecodeError`` text — ``Expecting ',' delimiter: line 1 column 154
(char 153)`` — which names neither what was lost nor what survived. The caller
cannot see the string it sent, so it re-sent it identically six times in one turn.
"""

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tool_search_validation import normalize_tool_call_entries

ARGS = {
    "project_slug": "shienga-agent-harness",
    "state": "needs-triage",
    "title": "OpenCode v2 migration - fleet on 2.0.30 verified end to end",
}

# Byte-exact reproduction of the live failure: the entry cut at the close of its
# `arguments` object, with `name` and every closing bracket missing.
TRUNCATED = '[{"arguments": ' + json.dumps(ARGS, separators=(", ", ": "))
TOOL = "mcp__testfoundry_issue_queue__get_issue"


def test_the_live_truncation_is_reproduced_byte_for_byte():
    """Lock the shape: 153 chars, and the exact error the live run reported."""
    assert len(TRUNCATED) == 153
    with pytest.raises(json.JSONDecodeError) as caught:
        json.loads(TRUNCATED)
    assert "line 1 column 154 (char 153)" in str(caught.value)


def test_truncated_calls_names_what_is_missing_and_echoes_the_arguments():
    entries, error = normalize_tool_call_entries(
        {"calls": TRUNCATED})
    assert entries == []
    assert "the entry's \"name\" and every closing bracket are missing" in error
    # The arguments survived, so the retry costs one call rather than a rethink.
    assert '"project_slug": "shienga-agent-harness"' in error
    assert "Re-send as" in error


def test_the_echoed_arguments_round_trip_into_a_working_retry():
    """The recovery is only useful if the caller can literally re-send it."""
    _, error = normalize_tool_call_entries({"calls": TRUNCATED})
    echoed = error.split("Your arguments survived verbatim: ", 1)[1].split(". Re-send as ", 1)[0]
    entries, retry_error = normalize_tool_call_entries(
        {"calls": [{"name": TOOL, "arguments": echoed}]})
    assert retry_error is None
    assert json.loads(echoed) == ARGS
    assert entries == [{"name": TOOL, "arguments": ARGS}]


def test_a_fenced_json_string_is_accepted():
    fenced = "```json\n" + json.dumps({"calls": [{"name": TOOL, "arguments": ARGS}]}) + "\n```"
    entries, error = normalize_tool_call_entries({"calls": fenced})
    assert error is None
    assert entries == [{"name": TOOL, "arguments": ARGS}]


def test_python_repr_quoting_is_accepted():
    """The other live variant: ``Expecting property name enclosed in double quotes``."""
    repr_string = str({"calls": [{"name": TOOL, "arguments": {"seq_no": 99012842}}]})
    entries, error = normalize_tool_call_entries({"calls": repr_string})
    assert error is None
    assert entries == [{"name": TOOL, "arguments": {"seq_no": 99012842}}]


def test_unrelated_malformation_reports_the_position_and_the_expected_shape():
    entries, error = normalize_tool_call_entries({"calls": '[{"name": ' + TOOL + ", }]"})
    assert entries == []
    assert "Near position" in error
    assert "Expected shape" in error


@pytest.mark.parametrize(
    "value",
    [
        [{"name": TOOL, "arguments": {"seq_no": 1}}],
        {"name": TOOL, "arguments": {"seq_no": 1}},
        json.dumps([{"name": TOOL, "arguments": {"seq_no": 1}}]),
    ],
)
def test_the_working_shapes_are_unchanged(value):
    entries, error = normalize_tool_call_entries({"calls": value})
    assert error is None
    assert entries == [{"name": TOOL, "arguments": {"seq_no": 1}}]
