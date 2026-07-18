# STDMS Knowledge Base (Phase 4 — dual RAG)

Documents are split into two embedding stores for higher precision:

| Folder | Store | Contents |
|--------|--------|----------|
| `space/` | `data/knowledge_index/rag_space.sqlite` | Satellite hardware/software, mission modes, eclipse, battery/EPS |
| `ground/` | `data/knowledge_index/rag_ground.sqlite` | Alert procedures, maintenance, operator / ground workflows |

Root-level `.md` files (if any) still index into the legacy `rag.sqlite` for compatibility.
Personal / offtopic files belong in `_archive/` (skipped).

**Rebuild** (air-gap, loopback Ollama embeddings):

```powershell
ollama pull nomic-embed-text
py -3.10 -m app.agent.rag.ingest
```

Or use **Rebuild Knowledge** on Home → AI Assistant (exports FSM mission modes into `space/mission_modes_auto.md`, then indexes both stores).

Search uses automatic segment routing: space vs ground vs both (anomaly/health questions).
