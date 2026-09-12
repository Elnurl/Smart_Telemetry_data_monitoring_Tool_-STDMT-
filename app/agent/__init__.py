"""Instrumentation Agent — Phase 0–4 tool chain, propose drafts, dual RAG, ToolHost API."""

from __future__ import annotations

from app.agent.audit import AgentAuditLog
from app.agent.bridge import AgentBridgeContext, ToolHost, attach_agent_bridge, stop_agent_bridge
from app.agent.function_calling import run_function_calling
from app.agent.loop import AgentMonitorLoop, AgentMonitorSignals
from app.agent.ollama_client import call_ollama, chat_ollama, ollama_reachable
from app.agent.log_monitor import LogAnalysis, LogEvent, LogMonitor, get_log_monitor
from app.agent.nodes import (
    CLLMNode,
    MLLMNode,
    MultiNodePipeline,
    RALLMNode,
    RLLMNode,
    heuristic_route,
)
from app.agent.prompts import (
    CHAT_PROMPT,
    CLLM_PROMPT,
    MLLM_PROMPT,
    RALLM_PROMPT,
    RLLM_PROMPT,
    SYSTEM_PROMPT,
    build_system_prompt,
)
from app.agent.runner import AgentRunner
from app.agent.server import create_app, start_agent_server
from app.agent.tools import invoke_tool, list_tools, search_knowledge, to_ollama_tools
from app.agent.rag import (
    DualStoreRetriever,
    ingest_dual_knowledge,
    ingest_knowledge_dir,
    search_knowledge_index,
)

__all__ = [
    "AgentAuditLog",
    "AgentBridgeContext",
    "AgentMonitorLoop",
    "AgentMonitorSignals",
    "AgentRunner",
    "CLLMNode",
    "CLLM_PROMPT",
    "CHAT_PROMPT",
    "LogAnalysis",
    "LogEvent",
    "LogMonitor",
    "MLLMNode",
    "MLLM_PROMPT",
    "MultiNodePipeline",
    "RALLMNode",
    "RALLM_PROMPT",
    "RLLMNode",
    "RLLM_PROMPT",
    "SYSTEM_PROMPT",
    "ToolHost",
    "attach_agent_bridge",
    "build_system_prompt",
    "call_ollama",
    "chat_ollama",
    "create_app",
    "DualStoreRetriever",
    "get_log_monitor",
    "heuristic_route",
    "ingest_dual_knowledge",
    "ingest_knowledge_dir",
    "invoke_tool",
    "list_tools",
    "ollama_reachable",
    "run_function_calling",
    "search_knowledge",
    "search_knowledge_index",
    "start_agent_server",
    "stop_agent_bridge",
    "to_ollama_tools",
]
