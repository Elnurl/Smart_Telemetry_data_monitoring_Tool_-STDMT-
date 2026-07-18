# STDMS v3.1 — Satellite Telemetry Data Monitoring Tool

PyQt desktop app: custom monitoring tabs, anomaly detection, fleet dashboard.

## License

MIT License — see [LICENSE](LICENSE).

## Instrumentation Agent (Phase 0–4) — air-gap

Local loopback API + Dashboard Agent Assistant with tool chains + RAG.

- Binds only to `127.0.0.1`
- Phase 0–3: snapshots, monitor loop, chat, function calling, propose_* + Approve/Reject
- Phase 4 RAG: put PDF/MD under `data/knowledge/`, then:
  ```powershell
  ollama pull nomic-embed-text
  py -3.10 -m app.agent.rag.ingest
  ```
  Or Home → Agent Assistant → **Rebuild Knowledge**. Index: `data/knowledge_index/rag.sqlite`
- `search_knowledge` returns local chunks + built-in product how-to
- Direct train / email / config apply remain blocked without human approval

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod http://127.0.0.1:8765/tabs
Invoke-RestMethod http://127.0.0.1:8765/tools
```

## Run (Python 3.10 only)

```powershell
cd STDMS_3.1
.\run.bat
```

Or: `py -3.10 main.py`

Do **not** use `python main.py` if default is Python 3.14.

## Setup

```powershell
py -3.10 -m pip install -r requirements.txt
copy data\email_config.example.json data\email_config.json
```

## Tests

```powershell
py -3.10 -m pytest tests/unit -q
```
