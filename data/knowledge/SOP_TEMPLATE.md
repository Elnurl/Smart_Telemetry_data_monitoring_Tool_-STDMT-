# STDMS SOP Word Template (agent reference)

When the operator asks to **create or update a procedure / SOP**, the agent must use the
**STANDARD OPERATING PROCEDURE (SOP)** Word template (`stdms_sop_word_v1`) and emit a `.docx`
via `propose_write_sop` / `propose_update_sop` (human Approve required).

## Layout

1. Title: **STANDARD OPERATING PROCEDURE (SOP)**
2. **General Information**
   - Process Title, Department, Contact Info
   - SOP ID, Effective Date, Revision Number
3. **Process Overview**
   - Process Description
   - Purpose & Scope
   - Definitions & Related Documents
4. **Process Steps** table: WBS | Task | Owner

## Tools

- `propose_write_sop` — create new SOP `.docx` under `data/knowledge/`
- `propose_update_sop` — replace an existing SOP `.docx`
- After Approve: file is written; run **Rebuild Knowledge** if RAG should index related MD notes

Do not invent a free-form layout for SOPs — always fill this template.
