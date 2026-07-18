# Alert Procedures (ground)

Ground / operator response when STDMS raises Warning, Critical, or OBS violations.

## Immediate steps
1. Confirm the monitoring tab name and last file / last update time on the Dashboard.
2. Open the tab → review fusion score, drift flag, OBS violations, and M-LLM log summary.
3. If the tab is in eclipse and M-LLM marked TCS warnings as mode-normal, do **not** escalate those as alerts.
4. For true anomalies: draft an alert for human Approve under Pending Agent Drafts.
5. If drift is detected, create or acknowledge a retrain signal; do not ignore repeated drift.

## Escalation
- OBS violated → notify the on-duty operator and propose_alert for approval.
- Critical health → prioritize; keep Propose→Approve (no silent auto-actions).

## References
- SOP-7 anomaly response sample
- Dashboard → Pending Agent Drafts / Pending Retrain Signals
