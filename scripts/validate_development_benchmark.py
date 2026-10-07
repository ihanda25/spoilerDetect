"""Validate development data, schema and saved predictions without inference."""
import hashlib
import json
import re
from pathlib import Path
from build_development_benchmark import validate
from evaluate_development_benchmark import invariants, summarize, THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/development-benchmark'


def check_schema(value, schema):
    """Small stdlib validator for the exact schema keywords used by this pack.

    Not a general JSON Schema engine. Fail closed on unimplemented keywords.
    """
    allowed = {'$schema', 'title', 'type', 'required', 'properties', 'enum', 'const',
               'allOf', 'if', 'then', 'items', 'minLength', 'pattern', 'minimum', 'exclusiveMinimum'}
    assert set(schema) <= allowed, set(schema)-allowed
    kinds = {'object': dict, 'array': list, 'string': str, 'null': type(None), 'number': (int, float)}
    if 'type' in schema:
        types = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        assert any(isinstance(value, kinds[t]) for t in types)
    if 'enum' in schema:
        assert value in schema['enum']
    if 'const' in schema:
        assert value == schema['const']
    if isinstance(value, dict):
        assert set(schema.get('required', [])) <= set(value)
        for key, sub in schema.get('properties', {}).items():
            if key in value:
                check_schema(value[key], sub)
    if 'items' in schema:
        for item in value:
            check_schema(item, schema['items'])
    if 'minLength' in schema:
        assert len(value) >= schema['minLength']
    if 'pattern' in schema:
        assert re.search(schema['pattern'], value)
    if 'minimum' in schema:
        assert value >= schema['minimum']
    if 'exclusiveMinimum' in schema:
        assert value > schema['exclusiveMinimum']
    for sub in schema.get('allOf', []):
        check_schema(value, sub)
    if 'if' in schema:
        try:
            check_schema(value, schema['if'])
        except AssertionError:
            pass
        else:
            check_schema(value, schema.get('then', {}))


def main():
    source = ROOT/'data/development-benchmark/examples.jsonl'
    rows = [json.loads(l) for l in source.read_text().splitlines()]
    validate(rows)
    invariants()
    schema = json.loads((OUT/'annotation.schema.json').read_text())
    for r in rows:
        check_schema(r, schema)
    responses = json.loads((OUT/'review-response-template.json').read_text())
    check_schema(responses, json.loads((OUT/'review-response.schema.json').read_text()))
    result = json.loads((OUT/'results.json').read_text())
    predictions = [json.loads(l) for l in (OUT/'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == len(predictions) == len(responses) == result['examples']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['examples_sha256']
    assert result['threshold'] == THRESHOLD
    for row, pred in zip(rows, predictions):
        assert row['id'] == pred['id'] and row['proposed_label'] == pred['proposed_label']
        assert pred['prediction'] == int(pred['score'] >= THRESHOLD)
        assert pred['visible_token_count'] == len(pred['visible_token_ids']) <= 512
        assert pred['removed_token_count'] == pred['full_token_count']-pred['visible_token_count']
    for surface, expected in result['synthetic_label_agreement'].items():
        selected = [p for p in predictions if p['proposed_label'] is not None and (surface == 'all' or p['surface'] == surface)]
        assert summarize([p['proposed_label'] for p in selected], [p['score'] for p in selected]) == expected
    proposals = json.loads((OUT/'assistant-proposals.json').read_text())
    by_id = {r['id']: r for r in rows}
    for p in proposals:
        span = p['assistant_evidence_span']
        if span:
            assert by_id[p['example_id']]['text'][span['start']:span['end']] == span['quote']
    assert result['unambiguous'] == sum(r['proposed_label'] is not None for r in rows)
    assert result['ambiguous_excluded'] == sum(r['proposed_label'] is None for r in rows)
    print(f'PASS: {len(rows)} schemas, saved score/threshold/length invariants, metrics, provenance hash, review templates, and evidence spans; no inference run.')


if __name__ == '__main__':
    main()
