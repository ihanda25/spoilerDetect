# Gemini 503 research — October 6, 2026

Most likely cause: temporary Google-side model serving/capacity pressure. Exact incident cannot be confirmed: our safe error logging stores exception class and HTTP code, not the server error message/details. No additional inference requests were made during research, and no secrets were inspected.

Google GenerateContent error reference describes 503 UNAVAILABLE as temporary overload/down/capacity shortage, while 429 indicates rate/quota limits, 403 permission issues, and 504 processing deadline failures. https://ai.google.dev/gemini-api/docs/generate-content/api-errors

Google forum guidance says 503 is unrelated to quota and varies with service demand/time. https://discuss.ai.google.dev/t/handling-429-503-errors-from-the-gemini-api/124640

September 21 report specifically includes gemini-3.8-flash, 3.7-flash, 3.6-flash on Free tier. A forum-team reply attributes these 503s to temporary overload. Historical evidence supports plausibility, not proof of an October 6 global outage. https://discuss.ai.google.dev/t/gemini-api-503/183913

Official status page checked live through browser: https://aistudio.google.com/status?tier=free — All Free tier systems are operational at research time. Page describes free traffic as sheddable capacity, billed traffic as critical priority. No reported current broad outage established. Model-specific or intermittent rejected requests remain possible despite green overall status.

Our client: public Google SDK, text-only generateContent, gemini-3.8-flash, low thinking, output cap 4096, temperature 1, serial requests spaced at least 15 seconds, 60-second HTTP timeout, SDK retries disabled to avoid double retries. Manual bounded 503-only retries (30/60 seconds) were exhausted. Some responses succeeded, so the secret/authentication/model route worked for those requests. We observed 503, not 429/403/504. No evidence that RAG or plot content caused the serving error. No retrieval or uploaded media is used.

Improvement options: randomized exponential backoff, a cooldown/circuit breaker after repeated 503s, preserving checkpoints rather than immediate endless restarts, or explicitly testing a different eligible free-tier Flash version in a separate output directory. Compare a tiny plain-text request and a representative excerpt across models before committing to a full run. Do not merge predictions across model versions into one benchmark result.

Google recommends bounded exponential backoff plus jitter and differentiating transient/server errors from client errors: https://ai.google.dev/gemini-api/docs/troubleshooting

Check this project's actual RPM/TPM/RPD in AI Studio before promising completion time. Limits depend on project and model, are per project rather than key, and capacity is not guaranteed: https://ai.google.dev/gemini-api/docs/rate-limits . A four-per-minute pace is not evidence that it is within this project's exact quota. The earlier 75–90 minute estimate omitted unknown daily quota and service availability. Creating a new key in the same project will not expand quota.

No paid upgrade is recommended as a presumed fix; it changes costs and is outside the user's no-purchases constraint. Batch inference has different eligibility/pricing and is not automatically a free alternative.

Retry implementation October 6: verified actual Free-tier quota 5 RPM / 250K TPM / 20 RPD in authenticated AI Studio project dashboard. Added 30-second gaps, bounded 503 waits 60/120/600 seconds plus jitter, one probe after the long cooldown, and five attempts maximum for this retry. Google documents daily reset at midnight Pacific.

Higher-volume free alternatives: Groq lists GPT-OSS 120B and 20B at 1,000 RPD, 30 RPM, 8K TPM and 200K TPD on Free plan; actual organization limits may vary. Recommended test GPT-OSS 120B using same blind paired evaluation, with token-aware pacing; do not mix predictions into the Gemini cache. https://console.groq.com/docs/rate-limits . OpenRouter lists 50 free requests/day and 20/minute without purchasing credits. https://openrouter.zendesk.com/hc/en-us/articles/39501163636379-OpenRouter-Rate-Limits-What-You-Need-to-Know . These are capacity options, not evidence of spoiler accuracy.
