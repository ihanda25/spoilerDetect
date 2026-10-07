"""Paired Gemini development diagnostic; API key stays inside Colab Secrets."""
import json
import random
from pathlib import Path
import time

from compare_three_models import sha, save, parse_answer
from compare_context_models import SYSTEM, load_inputs, load_annotations, report

MODEL = 'gemini-3.8-flash'
REQUEST_GAP_SECONDS = 30
RETRY_POLICY = '30s after each request; 503 retry waits 60/120s + jitter, then one 10min cooldown probe; quota/auth errors stop'


def retry_delay(attempt):
    """One long cooldown after the short retries; never an endless loop."""
    base = (60, 120, 600)[attempt]
    return base + random.uniform(0, base * .1)


def run(root, client, max_requests=5):
    from google.genai import types
    from importlib.metadata import version
    root = Path(root)
    rows, metadata, contexts = load_inputs(root/'blind.jsonl', root/'metadata.json', root/'contexts.json')
    annotations = load_annotations(root/'annotations.jsonl', {r['id'] for r in rows})
    out = root/'results'
    out.mkdir(exist_ok=True)
    config = dict(version=1, model=MODEL, dataset_sha256=sha(rows), metadata_sha256=sha(metadata),
                  contexts_sha256=sha(contexts), system=SYSTEM, temperature=1,
                  thinking_level='low', max_output_tokens=4096, sdk=version('google-genai'),
                  billing='user-confirmed free-tier project', retrieval=False)
    caches = {}
    for arm in ['gemini_title', 'gemini_context']:
        cfg = dict(config, arm=arm)
        path = out/(arm+'.json')
        cache = json.loads(path.read_text()) if path.exists() else dict(config=cfg, predictions={})
        if cache['config'] != cfg:
            raise ValueError('Cache configuration changed; use a new output directory')
        if not set(cache['predictions']) <= {r['id'] for r in rows}:
            raise ValueError('Unexpected cached IDs')
        for row in rows:
            pred = cache['predictions'].get(row['id'])
            if pred and pred['text_sha256'] != sha(row['text']):
                raise ValueError('Cached text changed')
        caches[arm] = cache
    requests = 0
    stopped = None
    previous = None
    started = time.monotonic()
    for row in rows:
        title = metadata[row['id']].get('work_title')
        context = contexts.get(title)
        for arm in ['gemini_title', 'gemini_context']:
            cache = caches[arm]
            if row['id'] in cache['predictions']:
                continue
            if arm == 'gemini_context' and not context:
                pred = dict(caches['gemini_title']['predictions'][row['id']], seconds=0,
                            reused_from='gemini_title', context_available=False)
                cache['predictions'][row['id']] = pred
                save(out/(arm+'.json'), cache)
                continue
            if previous is not None:
                time.sleep(max(0, REQUEST_GAP_SECONDS-(time.monotonic()-previous)))
            user = {'title': title, 'excerpt': row['text']}
            if arm == 'gemini_context':
                user.update(public_premise=context['premise'], plot_context=context['plot'])
            request_started = time.monotonic()
            for attempt in range(4):
                if requests >= max_requests:
                    stopped = {'reason': 'request_budget', 'max_requests': max_requests}
                    print('PAUSED: request budget reached; saved answers will resume later.', flush=True)
                    break
                requests += 1
                try:
                    response = client.models.generate_content(
                        model=MODEL, contents=json.dumps(user, ensure_ascii=False),
                        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=1,
                            max_output_tokens=4096, thinking_config=types.ThinkingConfig(thinking_level='low')))
                    raw = response.text or ''
                    break
                except Exception as exc:
                    # Never export exception messages, HTTP headers, client objects, or credentials.
                    code = getattr(exc, 'code', None)
                    if code == 503 and requests >= max_requests:
                        stopped = {'reason': 'request_budget', 'max_requests': max_requests, 'http_code': code}
                        print('PAUSED: request budget reached after service error; saved answers preserved.', flush=True)
                        break
                    if code == 503 and attempt < 3:
                        delay = retry_delay(attempt)
                        print('Temporary service error 503;', 'extended cooldown' if attempt == 2 else 'backoff',
                              'for', round(delay), 'seconds.', flush=True)
                        save(out/'status.json', dict(complete=False, phase='cooldown', http_code=503,
                             cooldown_seconds=delay, retry_attempt=attempt+1, row_id=row['id'], arm=arm,
                             predictions={a:len(c['predictions']) for a,c in caches.items()}, retry_policy=RETRY_POLICY))
                        time.sleep(delay)
                        continue
                    stopped = {'exception_type': type(exc).__name__, 'http_code': code}
                    print('STOPPED:', json.dumps(stopped), 'No model fallback or billing changes.', flush=True)
                    break
            previous = time.monotonic()
            if stopped:
                break
            label, valid = parse_answer(raw)
            usage = response.usage_metadata
            pred = dict(prediction=label, valid_output=valid, raw_output=raw,
                        text_sha256=sha(row['text']), seconds=time.monotonic()-request_started,
                        context_available=bool(context) if arm == 'gemini_context' else None,
                        dropped_tokens=0, model_version=getattr(response, 'model_version', None),
                        finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
                        input_tokens=getattr(usage, 'prompt_token_count', None),
                        output_tokens=getattr(usage, 'candidates_token_count', None),
                        thinking_tokens=getattr(usage, 'thoughts_token_count', None))
            cache['predictions'][row['id']] = pred
            save(out/(arm+'.json'), cache)
            save(out/'status.json', dict(complete=False, phase='running', rows=len(rows), requests_this_run=requests,
                 predictions={a:len(c['predictions']) for a,c in caches.items()}, retry_policy=RETRY_POLICY))
            print(arm, len(cache['predictions']), '/', len(rows), 'requests:', requests, flush=True)
        if stopped:
            break
    complete = all(len(c['predictions']) == len(rows) for c in caches.values())
    save(out/'status.json', dict(complete=complete, rows=len(rows), requests_this_run=requests,
         predictions={arm:len(c['predictions']) for arm,c in caches.items()}, stopped=stopped,
         elapsed_seconds=time.monotonic()-started, retry_policy=RETRY_POLICY, max_requests=max_requests))
    if complete:
        comparison = report(rows, metadata, contexts, annotations, caches)
        for arm, cache in caches.items():
            comparison['models'][arm]['reused_fallback_rows'] = sum(
                p.get('reused_from') == 'gemini_title' for p in cache['predictions'].values())
        baseline = json.loads((root/'baseline-comparison.json').read_text())
        assert baseline['models']['roberta']['config']['dataset_sha256'] == sha(rows)
        comparison['models']['roberta'] = baseline['models']['roberta']
        comparison['limitations'] = ['AI annotations are not human gold',
            'Only five labeled spoilers in real-content slice',
            'Context summaries are assistant-written, not human-verified',
            'Challenge and contexts share source plots',
            'Supplied context diagnostic, not a retrieval benchmark',
            'Gemini receives full context; Qwen previously had a 2048-token input cap']
        save(out/'comparison.json', comparison)
        print('GEMINI CONTEXT COMPARISON COMPLETE', flush=True)
    else:
        print('PARTIAL RUN SAVED; resume the same cell after quota/access is available.', flush=True)
    return complete
