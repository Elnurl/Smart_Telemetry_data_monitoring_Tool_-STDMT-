"""End-to-end smoke: dual RAG rebuild + eclipse logs + 4 multi-node asks."""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.agent.audit import AgentAuditLog
    from app.agent.bridge import MainWindowToolHost, set_tool_host
    from app.agent.log_monitor import LogMonitor, reset_log_monitor
    from app.agent.rag.ingest import ingest_dual_knowledge
    from app.agent.runner import AgentRunner
    from app.models.fsm import ensure_fsm_fields

    print("=== 1) Dual Rebuild Knowledge ===", flush=True)
    tabs_cfg = json.loads((ROOT / "data" / "custom_tabs_config.json").read_text(encoding="utf-8"))
    rebuild = ingest_dual_knowledge(tabs_config=tabs_cfg, force=False)
    print(
        f"ok={rebuild.get('ok')} docs={rebuild.get('documents')} chunks={rebuild.get('chunks')} "
        f"space={(rebuild.get('space') or {}).get('documents')} "
        f"ground={(rebuild.get('ground') or {}).get('documents')}",
        flush=True,
    )
    print(f"fsm={(rebuild.get('fsm_export') or {}).get('path')}", flush=True)
    if not rebuild.get("ok"):
        print("REBUILD_ERROR", rebuild.get("error") or rebuild.get("note"), flush=True)
        return 2

    print("=== 2) Eclipse mode + sample_mission.log ===", flush=True)
    reset_log_monitor()
    tab_id = "98dfc738-9a64-4148-997a-18d51f6dee70"
    cfg = ensure_fsm_fields(dict(tabs_cfg.get(tab_id) or {}))
    cfg["current_mission_mode"] = "eclipse"
    cfg["title"] = "Eclipse"
    monitor = LogMonitor(allow_llm=False)
    n = monitor.ingest_file(tab_id, ROOT / "data" / "sample_mission.log")
    analysis = monitor.analyze(tab_id, tab_config=cfg, apply_transitions=False, use_llm=False)
    print(
        f"ingested={n} filtered={analysis.false_positives_filtered} "
        f"anomalies={[a.get('source') for a in analysis.anomalies]}",
        flush=True,
    )
    print(f"summary={analysis.summary}", flush=True)

    class FakeTab:
        def __init__(self):
            self.tab_id = tab_id
            self.config = cfg
            self.last_mllm_summary = analysis.summary
            self.last_mllm_analysis = analysis.to_dict()
            self.last_snapshot = {
                "tab_id": tab_id,
                "title": "Eclipse",
                "health_state": "Warning",
                "mission_mode": "eclipse",
                "threshold_scale": 1.5,
                "drift": False,
                "obs_ok": True,
                "obs_mode_normal": 1,
                "mllm_summary": analysis.summary,
                "alert_count": 1,
                "trained_models": 2,
                "monitoring_active": True,
                "updated_at": "2026-07-18 20:30:00",
            }
            self.anomaly_events = deque(maxlen=50)

        def get_snapshot(self):
            return dict(self.last_snapshot)

    class FakeReg:
        def get_pending_retrain_signals(self):
            return []

        def create_draft_alert(self, **kwargs):
            return {"ok": True, "draft_id": "smoke-draft-1", **kwargs}

    class FakeWindow:
        def __init__(self):
            self.custom_tabs = {tab_id: FakeTab()}
            self.model_registry = FakeReg()
            self.tab_config_manager = type(
                "M", (), {"configs": tabs_cfg, "add_config": lambda *a, **k: True}
            )()

    host = MainWindowToolHost(FakeWindow())
    set_tool_host(host)
    audit = AgentAuditLog(ROOT / "data" / "agent_smoke_decisions.db")
    runner = AgentRunner(
        audit=audit,
        tool_host_getter=lambda: host,
        allow_llm=True,
        model="qwen3.5:9b",
    )

    questions = [
        "Why is Eclipse tab showing TCS warnings?",
        "What is the alert procedure for anomaly?",
        "Which tab needs attention first?",
        "Create tab from data/BatteryTemperature.csv",
    ]
    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n=== Q{i}: {q} ===", flush=True)
        out = runner.ask(q)
        reply = out.get("reply") or ""
        row = {
            "q": q,
            "route": out.get("route"),
            "node": out.get("node"),
            "agent_mode": out.get("agent_mode"),
            "llm_used": out.get("llm_used"),
            "tools": [t.get("tool") for t in (out.get("tool_trace") or []) if t.get("ok")],
            "reply": reply,
        }
        results.append(row)
        print(
            f"route={row['route']} node={row['node']} llm_used={row['llm_used']} tools={row['tools'][:12]}",
            flush=True,
        )
        print("REPLY:", flush=True)
        print(reply[:1500], flush=True)
        print("---", flush=True)

    r1, r2, r3, r4 = results
    checks = [
        ("Q1 route knowledge", r1["route"] == "knowledge"),
        ("Q1 node C-LLM", r1["node"] == "C-LLM"),
        ("Q1 mode-normal", "mode-normal" in r1["reply"].lower()),
        ("Q2 route knowledge", r2["route"] == "knowledge"),
        ("Q2 node C-LLM", r2["node"] == "C-LLM"),
        (
            "Q2 alert/sop content",
            any(
                k in r2["reply"].lower()
                for k in ("alert", "acknowledge", "approve", "operator", "drift", "obs")
            ),
        ),
        ("Q3 route tool", r3["route"] == "tool"),
        ("Q3 node RA-LLM", r3["node"] == "RA-LLM"),
        ("Q4 route tool", r4["route"] == "tool"),
        ("Q4 node RA-LLM", r4["node"] == "RA-LLM"),
        (
            "Q4 propose_create_tab once",
            sum(1 for t in r4["tools"] if t == "propose_create_tab") <= 1
            and (
                "propose_create_tab" in r4["tools"]
                or "approve" in r4["reply"].lower()
                or "draft" in r4["reply"].lower()
            ),
        ),
        (
            "Q4 create flow",
            any(
                t in r4["tools"]
                for t in ("inspect_data_folder", "suggest_tab_config", "propose_create_tab")
            )
            or "create" in r4["reply"].lower(),
        ),
    ]

    print("\n=== SMOKE CHECK SUMMARY ===", flush=True)
    ok_all = True
    for name, ok in checks:
        print(("PASS" if ok else "FAIL") + " - " + name, flush=True)
        ok_all = ok_all and ok
    print("OVERALL: " + ("PASS" if ok_all else "FAIL"), flush=True)
    (ROOT / "data" / "agent_smoke_results.json").write_text(
        json.dumps(
            {"checks": checks, "results": results, "overall": ok_all, "rebuild": {
                "ok": rebuild.get("ok"),
                "documents": rebuild.get("documents"),
                "chunks": rebuild.get("chunks"),
            }},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
