"""Phase 3 — multi-LLM node routing / FSM / RAG-only behavior."""

from __future__ import annotations

from app.agent.nodes import (
    CLLMNode,
    MLLMNode,
    MultiNodePipeline,
    RALLMContext,
    RALLMNode,
    RLLMNode,
    heuristic_route,
)
from app.agent.prompts import CLLM_PROMPT, RALLM_PROMPT, RLLM_PROMPT


def test_prompt_constants_present():
    assert "tool" in RLLM_PROMPT.lower()
    assert "knowledge" in RLLM_PROMPT.lower()
    assert "{log_summary}" in RALLM_PROMPT
    assert "{rag_context}" in CLLM_PROMPT


def test_rllm_routing():
    node = RLLMNode(allow_llm=False)
    assert node.route("create tab from data/x.csv") == "tool"
    assert node.route("what is the eclipse procedure?") == "knowledge"
    assert node.route("which tab needs attention?") == "tool"
    assert heuristic_route("Why is Eclipse tab showing warnings?") == "knowledge"


def test_rallm_fsm_context():
    node = RALLMNode(allow_llm=False)
    eclipse = node.reason(
        RALLMContext(
            query="Why warnings on Eclipse tab?",
            snapshot={"title": "Eclipse", "mission_mode": "eclipse", "health_state": "Warning"},
            log_summary="Eclipse entry detected. 1 WARNING filtered (mode-normal).",
            fsm_mode="eclipse",
            scale=1.5,
        ),
        host=None,
    )
    assert "mode-normal" in eclipse.reply.lower()
    assert eclipse.node == "RA-LLM"

    nominal = node.reason(
        RALLMContext(
            query="Why warnings?",
            snapshot={"title": "Battery", "mission_mode": "nominal", "health_state": "Warning"},
            log_summary="1 true anomaly: TCS WARNING.",
            fsm_mode="nominal",
            scale=1.0,
        ),
        host=None,
    )
    assert "anomal" in nominal.reply.lower()


def test_cllm_rag_only():
    node = CLLMNode(allow_llm=False)
    missing = node.answer(
        "What is the classified thruster secret code?",
        rag_context=[],
        fsm_context={"mode": "nominal", "threshold_scale": 1.0},
        log_summary="",
    )
    assert "knowledge bazasında yoxdur" in missing.lower() or "knowledge bazasinda yoxdur" in missing.lower()

    hits = [
        {
            "title": "SOP-7 anomaly response",
            "snippet": "Acknowledge the alert, isolate the channel, notify the on-call engineer.",
            "source": "SOP-7_anomaly_response.md",
        }
    ]
    found = node.answer(
        "What is the anomaly response procedure?",
        rag_context=hits,
        fsm_context={"mode": "nominal", "threshold_scale": 1.0},
    )
    assert "Acknowledge the alert" in found
    assert CLLMNode.MISSING not in found


def test_cllm_fsm_explains_eclipse_without_rag():
    node = CLLMNode(allow_llm=False)
    text = node.answer(
        "Why is Eclipse tab showing warnings?",
        rag_context=[],
        fsm_context={"mode": "eclipse", "threshold_scale": 1.5},
        log_summary="Eclipse entry detected. 3 WARNING filtered (mode-normal).",
    )
    assert "mode-normal" in text.lower()
    assert "eclipse" in text.lower()


def test_mllm_summarize_without_analysis():
    node = MLLMNode(allow_llm=False)
    text = node.summarize(fsm_state={"mode": "eclipse", "threshold_scale": 1.5})
    assert "eclipse" in text.lower()


def test_pipeline_routes_create_tab_to_ra(monkeypatch):
    pipe = MultiNodePipeline(allow_llm=False)

    class _Host:
        def list_tabs(self):
            return []

        def get_pending_retrain_signals(self):
            return []

    # Avoid RAG / host side effects
    monkeypatch.setattr(pipe.cllm, "_retrieve", lambda *a, **k: [])
    result = pipe.run("create tab from data/battery.csv", host=_Host())
    assert result.route == "tool"
    assert result.node == "RA-LLM"
    assert "[R-LLM] routing → tool" in result.reply


def test_pipeline_routes_why_eclipse_to_c(monkeypatch):
    pipe = MultiNodePipeline(allow_llm=False)
    monkeypatch.setattr(pipe.cllm, "_retrieve", lambda *a, **k: [])
    result = pipe.run(
        "Why is Eclipse tab showing warnings?",
        host=None,
    )
    assert result.route == "knowledge"
    assert result.node == "C-LLM"
    assert "[C-LLM]" in result.reply
