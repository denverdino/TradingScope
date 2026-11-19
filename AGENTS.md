# TradingScope Agent Guide

TradingScope uses AgentScope and DashScope Qwen for multi-agent trading analysis and post-market evaluation.
Use Python and dependency requirements from `pyproject.toml`, manage commands with `uv`, and do not introduce an OpenAI API dependency.

## Navigation

- Commands and quality checks: `Makefile`; dependency and pytest configuration: `pyproject.toml`.
- Models, debate limits, output directories, and vendor defaults: `tradingscope/default_config.py`; domain vocabulary: `GLOSSARY.md`.
- Analysis CLI and pipeline: `tradingscope/main.py`, `tradingscope/agents/workflow.py`; shared models and stage state: `tradingscope/agents/utils/context.py`.
- Output constraints: inspect the relevant analysis prompt in `tradingscope/agents/analysts/` (or the producing agent), then `tradingscope/agents/output.py` → `StructuredAgentRunner` in `tradingscope/agents/utils/structured_output.py` → `tradingscope/agents/renderers.py`; regressions: `tests/test_decision_policy_prompts.py`, `tests/test_output_models.py`, `tests/test_structured_agent_runner.py`, `tests/test_renderers.py`.
- Persistence and completed reads: `tradingscope/utils/oss_structured_output_uploader.py`, `tradingscope/utils/oss_structured_output_reader.py`; regression: `tests/test_structured_output_persistence.py`.
- Evaluation CLI requires `--tickers` (`make evaluate` does not supply it): `tradingscope/evaluate.py`; scoring and manifest-gated ingestion: `tradingscope/agents/evaluation/evaluator.py`, `tradingscope/agents/evaluation/oss_store.py`.
- Vendor routing and fallbacks: `tradingscope/dataflows/interface.py`; runtime configuration: `tradingscope/dataflows/config.py`. Tool preferences override category preferences, then routing fallbacks apply; fallbacks may be paid.
- Daily prices and Last Completed Session: `tradingscope/agents/utils/core_stock_tools.py`, `tradingscope/dataflows/y_finance.py`, `tradingscope/dataflows/stockstats_utils.py`, `tradingscope/agents/utils/context.py`; regressions: `tests/test_core_stock_tools.py`, `tests/test_technical_indicators_tools.py`, `tests/test_volume_analysis.py`.
- Options expiration, OI, and Max Pain semantics: `get_options_analysis` in `tradingscope/agents/utils/core_stock_tools.py` and `tradingscope/dataflows/y_finance.py`; regression: `tests/test_options_analysis.py`.
- Tool diagnostics: `tradingscope/agents/utils/local_audit.py`, `tradingscope/agents/utils/agent_utils.py`, `tradingscope/agents/utils/tool_response_helper.py`; structured-output diagnostics: `tradingscope/agents/utils/structured_output.py`.
- Offline guard and replay: `tests/conftest.py`, `tests/test_offline_network.py`, `tests/support/replay.py`, `tests/test_local_audit.py`, `tests/test_replay.py`.

## Credentials and External Effects

- Live analysis, evaluation, and paid or real-network integration tests require explicit user authorization; having credentials available is not authorization.
- For an authorized live application or integration-test command, source `~/daily/llm_env.sh` in the same shell as that command. Load any required local environment there too.
- Never print, inspect, persist, or commit loaded secrets, credentials, or tokens. Never commit `.env`, other local environment files, generated reports, or actual audit captures.
- `DASHSCOPE_API_KEY` authorizes model access; Finnhub company news uses `FINHUB_API_KEY`. Other provider, OSS, and email requirements live in their implementations; enable only the integrations the user requested.
- Do not push, open a pull request, upload reports or audits, or send email without explicit user authorization. Analysis can upload node artifacts immediately when OSS is configured; evaluation can fetch OSS data and send email even with `--dry-run`, which only skips recording evaluated keys.

## Offline Verification

- Pytest defaults to offline: real Requests, curl_cffi, HTTPX transport, socket connections, and DNS are blocked; mocked transports and local event-loop socketpairs remain usable.
- Live tests require all gates: an `integration` marker, explicit `--run-live`, `RUN_DASHSCOPE_INTEGRATION=1`, and the corresponding credentials. Ordinary tests remain offline even when credentials are present.
- Run focused offline tests with `RUN_DASHSCOPE_INTEGRATION=0 uv run pytest -q tests/test_foo.py`; consult `Makefile` for lint, formatting, and full-suite commands.
- `.github/workflows/ci.yml` runs offline quality checks on push/PR with read-only permissions and no secrets; `.github/workflows/trading-analysis.yml` is the live scheduled workflow, not a local verification shortcut.
- First reproduce a regression with a failing test, then implement and obtain fresh passing output. Mock external boundaries rather than bypassing network guards or structured validation.
- Format only task-owned files in a dirty worktree. Report exact checks and skipped verification; offline replay proves recorded behavior, not that a live model obeys a prompt.

## Structured Output and Atomic Completion

- JSON-producing workflow nodes use `StructuredAgentRunner`; failed schema or trading-rule validation stops downstream processing. Keep the analysis draft and structured conversion as distinct phases; retries must not regenerate the draft or refetch its evidence.
- `DashScopeResponseModel` must inline local Pydantic `$ref` definitions before sending structured-output tools: otherwise DashScope can serialize nested objects as JSON strings. Preserve Pydantic validation during normalization.
- Thinking-enabled requests use `tool_choice="auto"`; non-thinking structured-output requests use `tool_choice="required"`.
- Validated nodes are persisted as JSON and Markdown when they finish. Only after all nodes succeed does `persist_analysis_result()` write `full_report` and then `manifest.json` last.
- The manifest is the atomic completion marker. Consumers must use `fetch_completed_output()` or `async_fetch_completed_output()` and validate the manifest, never treat a lone node artifact as a completed run.
- A new run invalidates prior local/OSS manifests for that ticker/date; same-key runs are serialized within one process. Preserve both behaviors when changing persistence.
- CLI artifacts live under `<results_dir>/data/<date>/<ticker>/`; check `tradingscope/main.py` for `--output` behavior and `tradingscope/default_config.py` for directory defaults.

## Local Audit and Tracing

- Audit is opt-in: only explicitly setting `TOOL_AUDIT_DIR` creates local, per-run private JSONL. Leave it unset for no audit files; keep actual captures private and outside committed fixtures.
- `tradingscope/agents/utils/local_audit.py` redacts diagnostic copies; supported text arguments/results and visible model text are saved in full after redaction, not as truncated terminal previews. Execution inputs, outputs, cache keys, and validation remain unchanged.
- Audit does not capture raw HTTP requests or headers, download media, or persist binary payloads. Unsupported media has redacted metadata and is explicitly non-replayable; use synthetic fixtures for offline replay.
- Audit failures must emit only a fixed safe diagnostic, never fall back to printing raw content or interrupt normal analysis. See `tests/test_local_audit.py` and `tests/test_replay.py` for coverage.
- Tracing is opt-in via `TRACING_ENABLED`; provider/exporter lifecycle lives in `tradingscope/agents/utils/tracing.py`, middleware in `AgentContext`.
- Initialize the single global OpenTelemetry provider at the analysis/evaluation CLI boundary, not inside agents/workflows. Always call `shutdown_tracing()` in `finally` to flush and close it; isolate/mock the lifecycle when testing both CLIs in one process (`tests/test_tracing.py`).

## Working and Local Git Policy

- Inspect relevant implementation, tests, configuration, and current diff before editing. Preserve all unrelated changes in the dirty worktree; do not overwrite another worker's files or clean up unrelated code.
- Keep changes within the authorized scope; do not change global guidance, permissions, hooks, or tool configuration without explicit task authorization. `CLAUDE.md` remains the `@AGENTS.md` forwarding pointer.
- Never commit AI-generated design, specification, planning, or implementation-plan documents; keep them outside the repository or untracked for local use.
- Before merging a development branch into trunk (`main`/`master`), squash all branch commits into one; do not merge a multi-commit branch history directly.
- After changes are complete and verified, automatically create the squashed local commit under the existing standing authorization for local commits/merges, unless the user overrides it (including a no-commit task). This does not authorize pushes or other external effects.
- Finish with what changed, exact verification results, and remaining assumptions or skipped checks; do not claim unrun live-model or remote-CI verification.
