"""
HTML renderer for Instrumentation Agent chat (QTextBrowser-friendly).

Converts markdown-ish agent replies into styled bubbles with:
- Observation / Analysis / Recommendation sections
- Tool-name badges
- Status word badges (Healthy / Critical / Idle / …)
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Qt Rich Text supports a limited CSS subset — keep styles simple/inline.

_CHAT_CSS = """
body {
  background-color: #ffffff;
  color: #222222;
  font-family: 'Segoe UI', Arial, sans-serif;
  font-size: 15px;
  margin: 10px;
}
.msg-row { margin: 12px 0; }
.avatar {
  width: 30px;
  height: 30px;
  color: #ffffff;
  font-size: 13px;
  font-weight: bold;
  text-align: center;
  vertical-align: middle;
  background-color: #6e7781;
}
.bubble-user, .bubble-agent, .bubble-system, .bubble-thinking {
  background-color: #ffffff;
  color: #222222;
  padding: 10px 12px;
  border: 1px solid #d0d7de;
}
.bubble-thinking { font-style: italic; color: #555555; }
.meta {
  color: #555555;
  font-size: 12px;
  margin: 0 0 8px 0;
  padding: 0 0 4px 0;
}
.msg-body {
  color: #222222;
  margin: 0;
  padding: 0;
}
.section-title {
  color: #555555;
  font-size: 12px;
  font-weight: bold;
  letter-spacing: 1px;
  margin-top: 8px;
  margin-bottom: 4px;
}
.section-body { color: #222222; margin-bottom: 6px; font-size: 15px; }
.divider { color: #d0d7de; margin: 6px 0; }
.tools-label { color: #555555; font-size: 12px; }
.badge-tool, .badge-ok, .badge-fail,
.badge-healthy, .badge-critical, .badge-warning, .badge-idle, .badge-alert {
  background-color: #f0f2f5;
  color: #222222;
  padding: 1px 6px;
  font-size: 12px;
  font-family: Consolas, 'Courier New', monospace;
}
.empty-hint { color: #555555; font-style: italic; padding: 28px 8px; text-align: center; font-size: 15px; }
"""

_TOOLS_LINE_RE = re.compile(
    r"^\[tools:\s*(.+?)\]\s*\n?",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*([✓✗✔✖]|ok|fail)?", re.IGNORECASE)

_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?\[?\s*"
    r"(Observation|Analysis|Recommendation)"
    r"\s*\]?(?:\*\*)?\s*[:\-–]?\s*",
    re.IGNORECASE,
)

_STATUS_WORDS = (
    ("CRITICAL", "badge-critical"),
    ("WARNING", "badge-warning"),
    ("Healthy", "badge-healthy"),
    ("Critical", "badge-critical"),
    ("Warning", "badge-warning"),
    ("Idle", "badge-idle"),
    ("ALERT", "badge-alert"),
    ("Alert", "badge-alert"),
)

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def escape(text: str) -> str:
    return html.escape(text or "", quote=True)


def parse_tools_prefix(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Pull leading ``[tools: name✓, …]`` into structured badges; return rest."""
    raw = text or ""
    m = _TOOLS_LINE_RE.match(raw)
    if not m:
        return [], raw
    tools: List[Dict[str, Any]] = []
    for name, mark in _TOOL_TOKEN_RE.findall(m.group(1)):
        ok = True
        if mark:
            mk = mark.lower()
            ok = mk in ("✓", "✔", "ok")
        tools.append({"tool": name, "ok": ok})
    rest = raw[m.end() :]
    return tools, rest


def split_oar_sections(text: str) -> Optional[List[Tuple[str, str]]]:
    """
    Split into Observation / Analysis / Recommendation blocks.
    Returns None if fewer than 2 section headers found (fall back to body).
    """
    raw = (text or "").strip()
    if not raw:
        return None
    matches = list(_SECTION_RE.finditer(raw))
    if len(matches) < 2:
        return None
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip().title()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        sections.append((title, body))
    return sections or None


def inline_markdown(text: str) -> str:
    """Escape HTML then apply a small markdown subset + status badges."""
    if not text:
        return ""
        # Protect code spans first via placeholders
    codes: List[str] = []

    def _code_sub(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00CODE{len(codes) - 1}\x00"

    tmp = _MD_CODE.sub(_code_sub, text)
    tmp = escape(tmp)
    tmp = _MD_BOLD.sub(r"<b>\1</b>", tmp)
    tmp = _MD_ITALIC.sub(r"<i>\1</i>", tmp)
    for i, code in enumerate(codes):
        tmp = tmp.replace(
            f"\x00CODE{i}\x00",
            f"<code style=\"background-color:#f0f2f5;color:#222222;padding:1px 4px;\">{escape(code)}</code>",
        )
    # Status badges (after escape — match escaped plain words)
    for word, cls in _STATUS_WORDS:
        tmp = re.sub(
            rf"(?<![A-Za-z0-9_>]){re.escape(word)}(?![A-Za-z0-9_<])",
            f'<span class="{cls}">{word}</span>',
            tmp,
        )
    tmp = tmp.replace("\n", "<br/>")
    return tmp


def render_tool_badges(tools: Sequence[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    parts = ['<span class="tools-label">tools used:</span>&nbsp;']
    for t in tools:
        name = escape(str(t.get("tool") or "?"))
        ok = bool(t.get("ok", True))
        cls = "badge-tool badge-ok" if ok else "badge-tool badge-fail"
        mark = "✓" if ok else "✗"
        parts.append(f'<span class="{cls}">{name} {mark}</span>&nbsp;')
    return "".join(parts)


def render_body_html(text: str) -> str:
    """Render agent/user body: OAR sections when present, else inline markdown."""
    tools_from_text, rest = parse_tools_prefix(text)
    # tools_from_text is used by callers via parse_tools_prefix; body uses rest
    sections = split_oar_sections(rest)
    if not sections:
        return inline_markdown(rest)
    chunks: List[str] = []
    for i, (title, body) in enumerate(sections):
        if i:
            chunks.append('<div class="divider">────────────</div>')
        chunks.append(f'<div class="section-title">{escape(title.upper())}</div>')
        chunks.append(f'<div class="section-body">{inline_markdown(body)}</div>')
    return "".join(chunks)


def _avatar_cell(role: str) -> str:
    role_l = (role or "").lower()
    if role_l in ("you", "user", "operator"):
        glyph = "U"
    elif role_l in ("agent", "assistant"):
        glyph = "A"
    else:
        glyph = "S"
    bg = "#6e7781"
    return (
        f'<td class="avatar" bgcolor="{bg}" width="32" '
        f'style="background-color:{bg};color:#fff;text-align:center;'
        f'font-weight:bold;font-size:13px;">{glyph}</td>'
    )


def render_message_html(
    role: str,
    text: str,
    *,
    mode: Optional[str] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    ts: Optional[str] = None,
    thinking: bool = False,
) -> str:
    role_l = (role or "").strip().lower()
    # AI / system on the left; operator (You) on the right
    if role_l in ("you", "user", "operator"):
        bubble_cls = "bubble-user"
        align = "right"
        label = "You"
        avatar_role = "you"
    elif role_l in ("agent", "assistant"):
        bubble_cls = "bubble-thinking" if thinking else "bubble-agent"
        align = "left"
        label = f"Agent ({mode})" if mode else "Agent"
        avatar_role = "agent"
    else:
        bubble_cls = "bubble-system"
        align = "left"
        label = "System"
        avatar_role = "system"

    tools_list = list(tools or [])
    body_src = text or ""
    if not tools_list:
        parsed, body_src = parse_tools_prefix(body_src)
        tools_list = parsed
    else:
        # Caller already provided tools — still strip a leading tools line if present
        _, body_src = parse_tools_prefix(body_src)

    tools_html = render_tool_badges(tools_list) if tools_list else ""
    if thinking:
        body_html = "thinking…"
    else:
        sections = split_oar_sections(body_src)
        if sections:
            chunks: List[str] = []
            for i, (title, body) in enumerate(sections):
                if i:
                    chunks.append('<div class="divider">────────────</div>')
                chunks.append(f'<div class="section-title">{escape(title.upper())}</div>')
                chunks.append(f'<div class="section-body">{inline_markdown(body)}</div>')
            body_html = "".join(chunks)
        else:
            body_html = inline_markdown(body_src)

    meta = escape(label)
    if ts:
        meta = f"{escape(ts)} · {meta}"

    # QTextBrowser often collapses adjacent divs — use <br/> + nested table so
    # "You" / "Agent" never glue onto the prompt or reply text.
    tools_row = (
        f'<tr><td style="padding:0 0 6px 0;">{tools_html}</td></tr>'
        if tools_html
        else ""
    )
    bubble_style = "background-color:#ffffff;border:1px solid #d0d7de;color:#222222;"

    avatar_html = _avatar_cell(avatar_role)
    bubble_td = (
        f'<td class="{bubble_cls}" style="{bubble_style}padding:10px 12px;">'
        f'<table width="100%" cellspacing="0" cellpadding="0">'
        f'<tr><td class="meta" style="color:#555555;font-size:12px;padding:0 0 6px 0;">'
        f"{meta}</td></tr>"
        f"{tools_row}"
        f'<tr><td class="msg-body" style="color:#222222;padding-top:2px;">'
        f"{body_html}</td></tr>"
        f"</table>"
        f"</td>"
    )
    # Left side: avatar then bubble; right side: bubble then avatar
    inner_cells = (
        avatar_html + bubble_td if align == "left" else bubble_td + avatar_html
    )

    bubble = (
        f'<table width="100%" cellspacing="0" cellpadding="0" class="msg-row">'
        f'<tr><td align="{align}">'
        f'<table cellspacing="0" cellpadding="6" style="max-width:94%;">'
        f"<tr>{inner_cells}</tr></table></td></tr></table>"
    )
    return bubble


def render_chat_document(messages: Sequence[Dict[str, Any]]) -> str:
    """Full HTML document for QTextBrowser.setHtml."""
    if not messages:
        body = (
            '<div class="empty-hint">'
            "Chat history appears here. Ask a question or click Analyze Fleet."
            "</div>"
        )
    else:
        parts: List[str] = []
        for msg in messages:
            parts.append(
                render_message_html(
                    str(msg.get("role") or "system"),
                    str(msg.get("text") or ""),
                    mode=msg.get("mode"),
                    tools=msg.get("tools"),
                    ts=msg.get("ts"),
                    thinking=bool(msg.get("thinking")),
                )
            )
        body = "".join(parts)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_CHAT_CSS}</style></head>"
        f'<body style="background-color:#ffffff;color:#222222;font-size:15px;">{body}</body></html>'
    )


def messages_to_plain_transcript(messages: Sequence[Dict[str, Any]]) -> str:
    """Plain-text fallback for titles / legacy export."""
    lines: List[str] = []
    for msg in messages:
        role = str(msg.get("role") or "System").strip()
        if role.lower() in ("you", "user", "operator"):
            role = "You"
        elif role.lower() in ("agent", "assistant"):
            role = "Agent"
        else:
            role = "System"
        ts = str(msg.get("ts") or "").strip() or "--:--"
        text = str(msg.get("text") or "").strip()
        if msg.get("thinking"):
            text = "thinking..."
        tools = msg.get("tools") or []
        if tools:
            names = ", ".join(
                f"{t.get('tool')}{'✓' if t.get('ok', True) else '✗'}" for t in tools
            )
            text = f"[tools: {names}]\n{text}"
        mode = msg.get("mode")
        if role == "Agent" and mode:
            role = f"Agent ({mode})"
        if "\n" in text:
            lines.append(f"[{ts}] {role}:\n{text}")
        else:
            lines.append(f"[{ts}] {role}: {text}")
    return "\n".join(lines)


def migrate_transcript_to_messages(transcript: str) -> List[Dict[str, Any]]:
    """Best-effort parse of legacy plain transcripts into message dicts."""
    text = (transcript or "").strip()
    if not text:
        return []
    pattern = re.compile(
        r"(?:^|\n)\[(\d{1,2}:\d{2})\]\s+(You|Agent(?:\s*\([^)]*\))?|System)\s*:\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"role": "system", "text": text, "ts": ""}]
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        ts = m.group(1)
        role_raw = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        mode = None
        role_l = role_raw.lower()
        if role_l.startswith("agent"):
            role = "agent"
            mm = re.search(r"\(([^)]+)\)", role_raw)
            if mm:
                mode = mm.group(1).strip()
        elif role_l.startswith("you"):
            role = "you"
        else:
            role = "system"
        tools, rest = parse_tools_prefix(body)
        out.append(
            {
                "role": role,
                "text": rest,
                "ts": ts,
                "mode": mode,
                "tools": tools or None,
            }
        )
    return out
