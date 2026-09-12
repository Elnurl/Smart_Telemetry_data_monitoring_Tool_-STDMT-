# STDMS 4.0 Phase 1 — AI Engineering foundation

Chip Huyen *AI Engineering* üslubu: **ölç → sadə adaptasiya → yalnız lazım olanı əlavə et.**

Mövcud STDMS (anomaly + agent + dual RAG + propose/approve) saxlanılır.  
Bu fazanın məqsədi: *“AI işləyir”* əvəzinə **sübut olunmuş metriklər**.

```text
Task spec (A–F)
      ↓
evaluation JSONL datasets
      ↓
eval runner  ←  versioned prompts  ←  LLMProvider (Ollama | Test)
      ↓
metrics report (F1, recall@k, tool match)
```

---

## Uğur meyarı

- `py -3.10 -m pytest tests/evaluation -q` keçir
- `py -3.10 -m app.eval.runner --suite all --provider test` → hesabat
- Promptlar `prompts/` altında versiyalanır
- LLM çağırışı `LLMProvider` üzərindən (Ollama + TestProvider)

---

## Yeni fayl strukturu

```text
docs/ai-engineering/
  TASKS.md                 # Task A–F + qəbul meyarları
  PHASE1.md                # bu fayl

prompts/
  assistant/system_v1.txt
  diagnosis/system_v1.txt
  mllm/system_v1.txt

app/llm/
  base.py                  # LLMProvider protocol
  ollama_provider.py
  test_provider.py

app/eval/
  runner.py                # CLI
  metrics.py               # F1, recall@k, tool match
  loaders.py

data/evaluation/
  anomaly_cases.jsonl
  rag_cases.jsonl
  agent_cases.jsonl
  safety_cases.jsonl

tests/evaluation/
  test_eval_loaders.py
  test_eval_metrics.py
  test_eval_runner_smoke.py
```

---

## Task-lar (ölçülə bilən)

| ID | Task | Ölçü |
|----|------|------|
| **A** | Anomaly detection | Precision / Recall / F1 |
| **B** | Diagnosis + evidence | Evidence hit (keyword/rubric v1) |
| **C** | Knowledge retrieval | Recall@k, citation filename |
| **D** | Operational recommendation | Structured fields + safety flag |
| **E** | Action planning | Tool sequence; execute yalnız propose sonrası |
| **F** | Operator Q&A | Golden answer contains/exact (kiçik set) |

---

## Addımlar

1. **Task spec + dataset seed** — `TASKS.md` + `data/evaluation/*.jsonl` (20–30 synthetic EPS/battery case)
2. **Metrics + runner** — F1, recall@k, tool match; report → `reports/eval/latest.json`
3. **Prompt versioning** — `app/agent/prompts.py` → `prompts/*/system_v1.txt`; kod `load_prompt()`
4. **LLM gateway (nazik)** — `OllamaProvider` + `TestProvider`; eval offline işləsin
5. **Safety seed** — injection / “IGNORE… execute_*” → sistem execute etməməlidir

---

## Verify

```powershell
py -3.10 -m pytest tests/evaluation tests/unit/test_chat_attachments.py -q
py -3.10 -m app.eval.runner --suite all --provider test
```

---

## Bu planda yoxdur

- Fine-tuning
- Hybrid RAG + reranker (Phase 2)
- Satellite simulator / digital twin (Phase 6)
- Closed-loop ACT→verify (Phase 7)
- Slice C Wave 2 (`SecureAnomalyDetectionTool` 12k split)

---

## Niyə əvvəl bu?

STDMS artıq agent/RAG/approve ilə **güclü demo**dır.  
Kitaba görə ən böyük boşluq texnologiya yığını deyil — **evaluation + dataset + prompt/model abstraksiyası**.

Təsdiqləsən, icraya Phase 1 todo-ları ilə başlanılır.
