# Paused at user request

Do not run API calls or resume automatically. User plans to continue tomorrow.

Saved 8/100 RAG answers. All 100 baseline answers, frozen hybrid retrieval inputs, local vector index and locked labels are preserved. Latest run stopped on HTTP 429 before the pause. Inspect usage-budget.json and the saved rate-limit state when resuming; no quota-bypassing fallback or purchases.

Cumulative caps: 120 request attempts including retries, 100000 input/output tokens; errors with unknown usage count conservatively. The cap includes pre-guard attempts and survives restarts.

After user asks to resume, run `.venv/bin/python scripts/compare_qwen_locked_rag.py --run`. Completed answers are reused. If the caps or daily quota prevent completion, report a partial comparison; do not silently increase budgets.

No process or scheduled continuation is active. This is a paused partial experiment, not a final model comparison.
