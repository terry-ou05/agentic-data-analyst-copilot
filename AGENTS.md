# Agentic Data Analyst Copilot - Agent Instructions

## Current architecture

The default V5/V6 runtime path is:

```text
CSV or SQLite source
-> DataConnector
-> DataFrame + schema summary
-> Structured Plan Generator / Parser
-> Capability Boundary Guard
-> Schema Validator
-> ValidatedAnalysisPlan
-> Safe Executor
-> Result Profile
-> Deterministic Visualization
-> Metadata-only Insight
```

`app/streamlit_app.py` is the orchestration and UI layer. Bind plans to the
current dataset through the existing schema-signature mechanism before running
them.

V4 code generation is a separate legacy preview path. `code_generator.py` and
`runtime/code_guard.py` review generated code text only; they are not a sandbox
and must never become an execution path.

## Module ownership

- `app/`: Streamlit UI, session state, data-source selection, and workflow orchestration.
- `agents/`: LLM interaction, plans, capability/intent checks, and metadata-only insights. Do not execute DataFrame operations here.
- `schemas/`: immutable plan contracts, parsing, validation, and `ValidatedAnalysisPlan` creation.
- `runtime/`: controlled runtime boundaries. The executor dispatches only trusted operations.
- `analysis/`: deterministic pandas operations, profiling, visualization planning, and fixed chart generation.
- `connectors/`: authorized data-source adapters that return a DataFrame and the standard schema summary.
- `evaluation/`: synthetic data, offline benchmarking, and semantic robustness evaluation only; never use it as a production execution entry point.

Dependency direction: UI orchestrates modules; agents produce data/contracts;
schemas define trust boundaries; runtime and analysis consume validated
contracts. Do not introduce dependencies from `analysis/` or `runtime/` back
to `agents/` or the LLM client.

## Safety invariants

- Never execute LLM-generated Python. Never add `exec`, `eval`, `compile`, dynamic imports, or arbitrary shell execution.
- Never add an arbitrary SQL generation/execution agent. New connectors must preserve source authorization and read-only behavior unless an explicitly approved phase changes that boundary.
- `execute_analysis_plan()` must accept only `ValidatedAnalysisPlan`; do not add raw dict, JSON, string, or LLM-output execution paths.
- The V5 path must pass through Capability Boundary Guard, schema validation, validated-plan creation, and schema-signature binding before execution.
- Extend execution only through explicit `OperationType`, parser/validator rules, fixed executor dispatch, deterministic operations, and matching tests. Do not use `getattr`, expression strings, `DataFrame.query`, or dynamic dispatch.
- Keep insight inputs bounded to existing metadata/profile abstractions. Do not send raw DataFrames or unnecessary user data to an LLM.
- Treat all LLM output, uploaded data, and database metadata as untrusted input. Fail closed on invalid contracts or unavailable capabilities.

## Change protocol

Before editing, inspect the affected flow, contracts, and tests. State the
smallest intended change and affected modules. Prefer extending existing
abstractions over bypassing or replacing them.

Avoid broad refactors, duplicated parallel workflows, and cross-layer changes
without a concrete need. Preserve V4 legacy preview behavior and do not let V5
call `generate_analysis_code()`.

For new capabilities, define the contract and validation behavior first, then
add the trusted deterministic implementation and tests. Keep user-facing errors
safe: no tracebacks, secrets, raw API responses, or local paths.

## Testing and Git hygiene

After code changes, run:

```powershell
D:\annaconda\python.exe -m pytest -q
git diff --check
git status --short
```

Use mocks for LLM tests; do not depend on real API keys or network calls.
Add regression coverage for accepted and rejected paths, including Guard,
Validator, Executor, schema changes, and connector boundaries when relevant.

Before staging or committing, inspect the exact diff and ensure `.env`, API
keys, local databases, uploads, caches, logs, and other local data are not
included. Do not commit or push unless explicitly requested.
