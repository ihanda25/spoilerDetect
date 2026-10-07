"""Integrity tests for the context-model dev runner; no models or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest

from compare_context_models import (checked_cache, diagnostic_metrics, load_annotations, load_inputs,
                                    model_config, report)


class ContextComparisonIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'a', 'text': 'The hero returns.', 'surface': 'review', 'split': 'dev'}
        self.metadata = {'a': {'work_title': 'A Tale', 'group_id': 'g', 'slice': 'real_content_dev'}}
        self.contexts = {'A Tale': {'premise': 'A public setup.', 'plot': 'Later events.',
                                    'source_url': 'https://example.test/source',
                                    'additional_source_urls': ['https://example.test/extra']}}

    def test_context_files_validate_and_reject_unexpected_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'data').write_text(json.dumps(self.row)+'\n')
            (root/'metadata').write_text(json.dumps(self.metadata))
            (root/'contexts').write_text(json.dumps(self.contexts))
            self.assertEqual(len(load_inputs(root/'data', root/'metadata', root/'contexts')[0]), 1)
            (root/'metadata').write_text(json.dumps({'other': self.metadata['a']}))
            with self.assertRaises(ValueError):
                load_inputs(root/'data', root/'metadata', root/'contexts')

    def test_prediction_fingerprint_binds_context_but_not_labels(self):
        cfg = model_config('llm_context', [self.row], self.metadata, self.contexts,
                           'GPU', ('t', 'p'), 'Qwen', 'commit')
        changed = {'A Tale': {**self.contexts['A Tale'], 'plot': 'Changed plot.'}}
        self.assertNotEqual(cfg['contexts_sha256'], model_config(
            'llm_context', [self.row], self.metadata, changed, 'GPU', ('t', 'p'), 'Qwen', 'commit')['contexts_sha256'])
        with tempfile.TemporaryDirectory() as td:
            cache = {'config': cfg, 'predictions': {'a': {'text_sha256': __import__('compare_three_models').sha(self.row['text'])}}}
            path = Path(td)/'cache.json'
            path.write_text(json.dumps(cache))
            self.assertEqual(checked_cache(path, cfg, [self.row]), cache)
            with self.assertRaises(ValueError):
                checked_cache(path, {**cfg, 'revision': 'other'}, [self.row])
            cache['predictions']['a']['text_sha256'] = 'stale'
            path.write_text(json.dumps(cache))
            with self.assertRaises(ValueError):
                checked_cache(path, cfg, [self.row])

    def test_roberta_fingerprint_records_fixed_text_only_baseline(self):
        cfg = model_config('roberta', [self.row], self.metadata, self.contexts,
                           'GPU', ('t', 'p'), 'RoBERTa', 'revision')
        self.assertEqual(cfg['input'], 'excerpt_only')
        self.assertEqual(cfg['threshold'], 0.5)
        self.assertEqual(cfg['positive_index'], 1)
        self.assertEqual(cfg['max_input_tokens'], 512)
        self.assertEqual(cfg['dtype'], 'float32')
        self.assertIsNone(cfg['system'])

    def test_false_positive_rates_distinguish_abstentions(self):
        rows = [{'id': 'fp'}, {'id': 'abstain'}]
        annotations = {'fp': {'label': 0}, 'abstain': {'label': 0}}
        predictions = {'fp': {'prediction': 1, 'seconds': .2},
                       'abstain': {'prediction': None, 'seconds': .0}}
        result = diagnostic_metrics(rows, annotations, predictions)
        self.assertEqual(result['false_positive_rate_answered'], 1.0)
        self.assertEqual(result['false_positive_rate_including_abstentions'], .5)
        self.assertEqual(result['inference_seconds'], .2)

    def test_annotations_require_ai_dev_and_context_fallback_is_counted(self):
        row = self.row
        annotation = {'id': 'a', 'split': 'dev', 'label_provenance': 'ai', 'label': 1}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'ann.jsonl'
            p.write_text(json.dumps(annotation)+'\n')
            labels = load_annotations(p, {'a'})
            cfg = {'x': 1}
            caches = {arm: {'config': cfg, 'predictions': {'a': {
                'prediction': None, 'seconds': 0.0, 'source_inference_seconds': .1,
                'reused_from': 'llm_title', 'context_available': False}}}
                for arm in ('roberta', 'llm_title', 'llm_context')}
            result = report([row], {'a': {'work_title': 'Missing', 'slice': 's'}}, {}, labels, caches)
            m = result['models']['llm_context']['metrics']['all']
            self.assertEqual(m['abstained_positives'], 1)
            self.assertEqual(m['recall_including_abstentions'], 0)
            self.assertEqual(m['per_slice']['s']['total'], 1)
            self.assertEqual(m['inference_seconds'], 0.0)
            self.assertEqual(result['models']['llm_context']['inference_seconds'], 0.0)
            self.assertEqual(result['models']['llm_context']['fallback_source_inference_seconds'], .1)
            self.assertEqual(result['models']['llm_context']['reused_fallback_rows'], 1)
            annotation['label_provenance'] = 'human'
            p.write_text(json.dumps(annotation)+'\n')
            with self.assertRaises(ValueError):
                load_annotations(p, {'a'})


if __name__ == '__main__':
    unittest.main()
