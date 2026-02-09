"""Analyze a session JSONL file for empty messages."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/nibot/.nibot/workspace/sessions/web_web_a4767aee.jsonl"

with open(path) as f:
    lines = [json.loads(l) for l in f if l.strip()]

msgs = [m for m in lines if not m.get("_type")]
empty = []

for i, m in enumerate(msgs):
    role = m.get("role", "?")
    content = m.get("content", "") or ""
    has_tc = bool(m.get("tool_calls"))
    is_empty = not content.strip()
    tag = " <<< EMPTY" if is_empty else ""
    tc_tag = " [+tool_calls]" if has_tc else ""
    preview = content[:80].replace("\n", "\\n") if content.strip() else "(EMPTY)"
    print(f"{i:3d} {role:10s}{tc_tag:16s} len={len(content):5d} | {preview}{tag}")
    if is_empty:
        empty.append((i, role, has_tc))

print(f"\n=== Summary ===")
print(f"Total messages: {len(msgs)}")
print(f"Empty messages: {len(empty)}")
print(f"\nEmpty breakdown:")
for idx, role, has_tc in empty:
    reason = "tool_call wrapper (LLM called tool without text)" if has_tc else "empty content"
    print(f"  msg[{idx:2d}] role={role:10s} tool_calls={has_tc} => {reason}")
