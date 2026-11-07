import pytest

from agent.message_metadata import (
    INFLIGHT_TASK_RESTATEMENT, TYPED_ASK, append_message, is_compaction_scaffolding, record_absorbed_message,
    replay_metadata_fields, stamp_message_timestamp,
)


def test_stamp_preserves_source_timestamp(monkeypatch):
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: 999.0)
    message = {"role": "user", "content": "hello", "timestamp": 123.0}

    result = stamp_message_timestamp(message)

    assert result is message
    assert message["timestamp"] == 123.0


def test_append_stamps_same_mapping_at_append_time(monkeypatch):
    monkeypatch.setattr("agent.message_metadata.wall_time", lambda: 456.0)
    messages = []
    message = {"role": "tool", "content": "ok"}

    result = append_message(messages, message)

    assert result is message
    assert messages == [message]
    assert message["timestamp"] == 456.0


def test_replay_keeps_a_user_rows_typed_ask_beside_the_scaffolding_flag():
    """Gateway replay carries the stamp (previous_message reads it after a restart) without
    turning the row into compaction scaffolding, which would skip its timestamp."""
    stamped = {"role": "user", "content": "x", "display_metadata": {TYPED_ASK: "ask", "other": 1}}
    both = {"role": "user", "content": "x", "display_metadata": {TYPED_ASK: "ask", INFLIGHT_TASK_RESTATEMENT: True}}

    assert replay_metadata_fields(stamped) == {"display_metadata": {TYPED_ASK: "ask"}}
    assert not is_compaction_scaffolding(stamped)
    assert replay_metadata_fields(both) == {"display_metadata": {INFLIGHT_TASK_RESTATEMENT: True, TYPED_ASK: "ask"}}
    assert replay_metadata_fields({**stamped, "role": "assistant"}) == {}


_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def _stamped_image_turn():
    return {"role": "user", "content": [{"type": "text", "text": "what is this"}, _IMAGE,
                                        {"type": "text", "text": "PLUGIN NOTE"}], "display_metadata": {TYPED_ASK: "what is this"}}


def _previous(row):
    from agent.turn_context import _previous_user_message_text
    return _previous_user_message_text([row, {"role": "assistant", "content": "x"}])


def test_a_summary_carrier_restating_the_in_flight_turn_takes_its_typed_ask():
    """Compaction restates an unfinished stamped image turn onto the summary carrier; the carrier's
    only ask is that turn's, so ``previous_message`` must not read its notes or the restatement."""
    from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX, _SUMMARY_END_MARKER

    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.quiet_mode = True
    carrier = {"role": "user", "content": f"{SUMMARY_PREFIX}\nold work\n{_SUMMARY_END_MARKER}", "_compressed_summary": True}
    compressed = compressor._reappend_inflight_user_task([carrier], _stamped_image_turn())

    assert compressed == [carrier]  # merged onto the carrier: the transcript already ends on a user row
    assert carrier["display_metadata"][TYPED_ASK] == "what is this"
    assert _previous(carrier) == "what is this"


def test_an_anchor_leading_a_scaffolding_turn_keeps_its_typed_ask():
    from agent.conversation_compression import _merge_anchor_into_user_message
    from tools.todo_tool import TODO_INJECTION_HEADER

    target = {"role": "user", "content": f"{TODO_INJECTION_HEADER}\n- [ ] a", "_todo_snapshot_synthetic": True}
    _merge_anchor_into_user_message(target, _stamped_image_turn())

    assert _previous(target) == "what is this"


def _summary(flagged):
    from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER
    return {"role": "user", "content": f"{SUMMARY_PREFIX}\nold work\n{_SUMMARY_END_MARKER}",
            **({"_compressed_summary": True} if flagged else {})}


def _compressor():
    from agent.context_compressor import ContextCompressor
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.quiet_mode = True
    return compressor


@pytest.mark.parametrize("ends_on_user", [True, False], ids=["merged-onto-carrier", "standalone-restatement"])
def test_a_turn_still_in_flight_across_two_compactions_keeps_its_typed_ask(ends_on_user):
    """The second compaction's in-flight row is the first one's carrier (it holds the restated task and
    its stamp); restating it again, merged or standalone, must keep the ask, not the notes or header."""
    first = _summary(True)
    _compressor()._reappend_inflight_user_task([first], _stamped_image_turn())
    second = _summary(True) if ends_on_user else {**_summary(True), "role": "assistant"}
    compressed = _compressor()._reappend_inflight_user_task([second], first)

    restated = compressed[-1]
    assert (restated is second) is ends_on_user
    assert restated["display_metadata"][TYPED_ASK] == "what is this"
    assert _previous(restated) == "what is this"


@pytest.mark.parametrize("flagged", [True, False], ids=["flagged", "content-only"])
def test_a_summary_folded_under_a_stamped_row_adds_no_ask(flagged):
    """The joined row is not a carrier on the read path, so dropping the stamp would read the summary."""
    from agent.agent_runtime_helpers import _merge_consecutive_users

    survivor = {"role": "user", "content": "ask\n[screenshot]\n\nPLUGIN NOTE", "display_metadata": {TYPED_ASK: "ask"}}
    merged, _ = _merge_consecutive_users([survivor, _summary(flagged), {"role": "assistant", "content": "x"}])

    assert merged[0]["display_metadata"][TYPED_ASK] == "ask"
    assert _previous(merged[0]) == "ask"


def test_an_unflagged_carrier_takes_the_ask_it_absorbs():
    carrier = _summary(False)
    record_absorbed_message(carrier, _stamped_image_turn())

    assert carrier["display_metadata"][TYPED_ASK] == "what is this"
