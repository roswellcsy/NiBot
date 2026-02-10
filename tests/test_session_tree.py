"""Session tree branching tests (Phase 1)."""
from __future__ import annotations

from nibot.session import Session


def test_add_message_returns_id():
    """add_message returns a 12-char hex string."""
    s = Session(key="t1")
    mid = s.add_message("user", "hello")
    assert isinstance(mid, str)
    assert len(mid) == 12
    assert all(c in "0123456789abcdef" for c in mid)


def test_messages_have_id_parent_id():
    """Messages contain id and parent_id fields."""
    s = Session(key="t2")
    m1 = s.add_message("user", "first")
    m2 = s.add_message("assistant", "reply")
    assert s.messages[0]["id"] == m1
    assert s.messages[0]["parent_id"] == ""  # first message has no parent
    assert s.messages[1]["id"] == m2
    assert s.messages[1]["parent_id"] == m1  # auto-linked to previous


def test_get_branch_linear():
    """Linear chain: get_branch(leaf) == get_history() content."""
    s = Session(key="t3")
    s.add_message("user", "a")
    m2 = s.add_message("assistant", "b")
    branch = s.get_branch(m2)
    history = s.get_history()
    # Same content, same order
    assert len(branch) == len(history)
    for b, h in zip(branch, history):
        assert b["role"] == h["role"]
        assert b["content"] == h["content"]


def test_get_branch_forked():
    """Forked tree: different leaves yield different paths."""
    s = Session(key="t4")
    root = s.add_message("user", "question")
    # Branch A: root -> a1 -> a2
    a1 = s.add_message("assistant", "answer-A", parent_id=root)
    a2 = s.add_message("user", "followup-A", parent_id=a1)
    # Branch B: root -> b1 -> b2
    b1 = s.add_message("assistant", "answer-B", parent_id=root)
    b2 = s.add_message("user", "followup-B", parent_id=b1)

    branch_a = s.get_branch(a2)
    branch_b = s.get_branch(b2)

    assert [m["content"] for m in branch_a] == ["question", "answer-A", "followup-A"]
    assert [m["content"] for m in branch_b] == ["question", "answer-B", "followup-B"]


def test_get_branch_backward_compat():
    """Old messages without id: get_branch() falls back to get_history()."""
    s = Session(key="t5")
    # Simulate old-format messages (no id/parent_id)
    s.messages.append({"role": "user", "content": "old msg"})
    s.messages.append({"role": "assistant", "content": "old reply"})

    branch = s.get_branch("")  # empty leaf
    assert len(branch) == 2
    assert branch[0]["content"] == "old msg"

    branch2 = s.get_branch("nonexistent")
    assert len(branch2) == 2  # no ids on messages -> fallback


def test_get_history_excludes_id_parent_id():
    """get_history() strips id and parent_id from output."""
    s = Session(key="t6")
    s.add_message("user", "hello")
    history = s.get_history()
    assert "id" not in history[0]
    assert "parent_id" not in history[0]
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"


def test_add_message_backfills_legacy_parent():
    """Legacy messages without id get backfilled so new messages can link."""
    s = Session(key="t7")
    # Simulate old-format messages (no id/parent_id)
    s.messages.append({"role": "user", "content": "legacy msg 1"})
    s.messages.append({"role": "assistant", "content": "legacy reply"})

    # New message should backfill the last legacy message's id
    new_id = s.add_message("user", "new question")
    assert s.messages[-1]["parent_id"] != ""  # linked to backfilled id
    assert s.messages[-2].get("id", "").startswith("_legacy_")

    # parent link exists
    assert s.messages[-1]["parent_id"] == s.messages[-2]["id"]
    # get_branch traces back through backfilled legacy id
    branch = s.get_branch(new_id)
    # Should include at least the backfilled parent and the new message
    assert len(branch) >= 2
    assert branch[-1]["content"] == "new question"


def test_compacted_summary_field():
    """Session has compacted_summary field, empty by default."""
    s = Session(key="t8")
    assert s.compacted_summary == ""
    s.compacted_summary = "Previously discussed X and Y."
    assert s.compacted_summary == "Previously discussed X and Y."
