# SOP-7 — Anomaly Response (Sample)

This is a **sample** department procedure for STDMS RAG testing. Replace with real NV/department SOPs.

## When anomaly or Warning health is detected

1. Confirm the monitoring tab name and last file / last update time on the Dashboard.
2. Open the tab → review fusion score, drift flag, and OBS violations.
3. If drift is detected, create or acknowledge a retrain signal; do not ignore repeated drift.
4. If OBS is violated, escalate to the on-duty operator and draft an alert for approval.
5. Do not retrain production models without human approval in Pending Retrain Signals / Agent Drafts.

## Retrain guidance

- Prefer propose/acknowledge workflow; never silent auto-train in air-gap ops.
- After retrain, run one On-Demand monitoring cycle and compare health to the previous Warning state.

## References

- STDMS custom tab: Monitoring Controls → Start Monitoring
- Agent Assistant: Ask / Analyze Fleet for tool-backed briefing
