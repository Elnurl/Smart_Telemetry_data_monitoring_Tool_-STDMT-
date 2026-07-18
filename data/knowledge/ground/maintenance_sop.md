# Maintenance SOP (ground)

## Model / monitoring maintenance
1. Prefer propose/acknowledge workflow; never silent auto-train in air-gap ops.
2. After retrain, run one On-Demand monitoring cycle and compare health to the previous Warning state.
3. After confirmed maintenance windows, operators may rebaseline drift reference on the tab.
4. Keep department SOPs under `data/knowledge/ground/`; satellite hardware notes under `data/knowledge/space/`.
5. After adding or editing docs, click **Rebuild Knowledge** so dual RAG indexes refresh.

## Operator checklist
- Start Monitoring is per-tab (Monitoring Controls), not Dashboard Ask.
- Approve pending drafts before train/start/alert actions execute.
