"""Unit tests for Instrumentation Agent HTML chat renderer."""

from app.agent.chat_render import (
    inline_markdown,
    migrate_transcript_to_messages,
    parse_tools_prefix,
    render_chat_document,
    render_message_html,
    split_oar_sections,
)


def test_inline_markdown_bold_and_code():
    html = inline_markdown("**Alert ID:** `42` and *note*")
    assert "<b>Alert ID:</b>" in html
    assert "<code" in html and "42" in html
    assert "<i>note</i>" in html
    assert "**" not in html


def test_status_badges():
    html = inline_markdown("Tab is Healthy, one Critical, rest Idle")
    assert "Healthy" in html and "Critical" in html and "Idle" in html


def test_parse_tools_prefix():
    tools, rest = parse_tools_prefix("[tools: list_watchlist✓, get_fleet_status✗]\nHello")
    assert len(tools) == 2
    assert tools[0]["tool"] == "list_watchlist" and tools[0]["ok"] is True
    assert tools[1]["tool"] == "get_fleet_status" and tools[1]["ok"] is False
    assert rest.startswith("Hello")


def test_split_oar_sections():
    text = (
        "[Observation] Fleet has 3 tabs.\n"
        "[Analysis] One is Critical due to drift.\n"
        "[Recommendation] Approve propose_alert."
    )
    sections = split_oar_sections(text)
    assert sections is not None
    titles = [t for t, _ in sections]
    assert titles == ["Observation", "Analysis", "Recommendation"]
    assert "Critical" in sections[1][1]


def test_render_message_agent_oar_and_tools():
    html = render_message_html(
        "agent",
        "[Observation] A\n[Analysis] B\n[Recommendation] C",
        mode="LLM",
        tools=[{"tool": "list_watchlist", "ok": True}],
        ts="16:06",
    )
    assert "OBSERVATION" in html
    assert "ANALYSIS" in html
    assert "RECOMMENDATION" in html
    assert "list_watchlist" in html
    assert 'align="left"' in html


def test_render_message_user_bubble():
    html = render_message_html("you", "Give me a full health report", ts="16:05")
    assert "Give me a full health report" in html
    assert 'align="right"' in html
    # Label and body must not concatenate (was "YouGive" / "YouCrea")
    assert "YouGive" not in html.replace(" ", "")
    assert 'class="meta"' in html or "font-size:12px" in html
    assert "msg-body" in html


def test_render_message_agent_on_left():
    html = render_message_html("agent", "Hello", mode="LLM", ts="16:06")
    assert 'align="left"' in html
    assert "Hello" in html
    assert "Agent (LLM)Hello" not in html.replace(" ", "").replace("\n", "")
    assert "msg-body" in html


def test_render_chat_document_empty():
    doc = render_chat_document([])
    assert "Ask a question" in doc
    assert "font-size:15px" in doc or "#ffffff" in doc


def test_migrate_legacy_transcript():
    transcript = (
        "[16:05] You: Hello fleet\n"
        "[16:06] Agent (LLM):\n"
        "[tools: get_fleet_status✓]\n"
        "[Observation] OK\n[Analysis] Fine\n[Recommendation] None"
    )
    msgs = migrate_transcript_to_messages(transcript)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "you"
    assert msgs[1]["role"] == "agent"
    assert msgs[1].get("mode") == "LLM"
    assert msgs[1].get("tools")
