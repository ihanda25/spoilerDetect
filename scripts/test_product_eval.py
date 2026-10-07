"""Lightweight stdlib tests. No torch, model loading, network, or paid inference.

Run: python -m unittest discover -s scripts -p test_product_eval.py -v
"""
import contextlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import product_eval as pe


def row(rid, split='dev', label=0, **kwargs):
    return dict(id=rid, text='Unique text '+rid, surface='comment', group_id=rid,
                split=split, label=label, label_provenance='human', review_status='human_reviewed',
                reviewer_id='unit-test-fixture-person', reviewed_at='2026-10-05T12:00:00Z', **kwargs)


class ProductEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = [row('d0'), row('d1', label=1), row('t0', 'test'), row('t1', 'test', 1)]
        self.data = self.root/'data.jsonl'
        self.save_data()

    def tearDown(self):
        self.temp.cleanup()

    def save_data(self):
        self.data.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def cache(self, name, split, scores, config=None, selection=None):
        config = config or {'mode':'truncate', 'model':'fake-for-unit-test'}
        header = pe.cache_header(pe.file_hash(self.data), config, split, pe.digest(selection) if selection else None)
        predictions = {}
        for r, score in zip([r for r in self.rows if r['split'] == split], scores):
            predictions[r['id']] = dict(text_sha256=pe.digest(r['text']), score=score, windows=[dict(score=score)])
        path = self.root/name
        pe.atomic_write(path, pe.seal(dict(header=header, predictions=predictions)))
        return path

    def test_window_boundaries_and_complete_coverage(self):
        self.assertEqual(pe.window_ranges(0, 510, 128, 'window'), [(0, 0)])
        self.assertEqual(pe.window_ranges(510, 510, 128, 'window'), [(0, 510)])
        self.assertEqual(pe.window_ranges(511, 510, 128, 'window'), [(0, 510), (382, 511)])
        self.assertEqual(pe.window_ranges(1000, 510, 128, 'truncate'), [(0, 510)])
        spans = pe.window_ranges(100000, 510, 128, 'window')
        self.assertGreater(len(spans), 100)
        self.assertEqual(spans[-1][1], 100000)
        self.assertEqual(set(i for a,b in spans for i in range(a,b)), set(range(100000)))
        self.assertTrue(all(b-a <= 510 for a,b in spans))
        self.assertTrue(all(spans[i-1][1]-spans[i][0] == 128 for i in range(1,len(spans))))
        for overlap in (-1,510,511):
            with self.assertRaises(ValueError):
                pe.window_ranges(1000,510,overlap,'window')

    def test_aggregation_and_invalid_scores(self):
        self.assertEqual(pe.aggregate([.01,.99,.1]), .99)
        for scores in ([], [math.nan], [math.inf], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                pe.aggregate(scores)

    def test_threshold_endpoints_ties_and_comparator(self):
        threshold, result = pe.select_threshold([0,1],[0.,1.])
        self.assertEqual(threshold,1.)
        self.assertEqual(result['f1'],1.)
        self.assertEqual(pe.select_threshold([0,1],[0.,0.])[0],0.)
        self.assertEqual(pe.select_threshold([0,1],[1.,1.])[0],1.)
        # Equal F1 at .9 and .7: highest cutoff wins.
        self.assertEqual(pe.select_threshold([1,0,0,1],[.9,.8,.8,.7])[0],.9)
        self.assertEqual(pe.metrics([0,1],[0.,1.],math.nextafter(1.,math.inf))['tp'],0)
        self.assertEqual(pe.metrics([1],[.5],.5)['tp'],1)
        self.assertEqual(pe.metrics([1],[math.nextafter(.5,0)],.5)['fn'],1)
        for labels,scores in (([],[]),([0],[.3]),([1],[.3]),([True,0],[.3,.2])):
            with self.assertRaises(ValueError):
                pe.select_threshold(labels,scores)

    def test_threshold_matches_bruteforce(self):
        rng=random.Random(23)
        for _ in range(80):
            labels=[0,1]+[rng.randrange(2) for _ in range(30)]
            scores=[rng.choice([0.,.1,.3,.7,1.]) for _ in labels]
            candidates=set(scores)|{0.,math.nextafter(1.,math.inf)}
            expected=max((pe.metrics(labels,scores,t)['f1'],t) for t in candidates)
            actual,m=pe.select_threshold(labels,scores)
            self.assertEqual((m['f1'],actual), expected)

    def test_group_and_text_leakage(self):
        pe.load_data(self.data)
        self.rows[2]['group_id']='d0'
        self.save_data()
        with self.assertRaisesRegex(ValueError,'Group leakage'):
            pe.load_data(self.data)
        self.rows[2]['group_id']='t0'
        self.rows[2]['text']='  UNIQUE   TEXT D0 '
        self.save_data()
        with self.assertRaisesRegex(ValueError,'text leakage'):
            pe.load_data(self.data)

    def test_synthetic_pending_corpus_policy(self):
        r=row('x')
        self.assertTrue(pe.eligible(r,'human'))
        r['review_status']='pending_human_review'
        self.assertFalse(pe.eligible(r,'human'))
        r.update(label_provenance='corpus',review_status='corpus_labeled')
        self.assertFalse(pe.eligible(r,'human'))
        self.assertTrue(pe.eligible(r,'corpus'))
        r.update(label_provenance='synthetic',review_status='human_reviewed')
        self.rows[0]=r
        self.save_data()
        with self.assertRaisesRegex(ValueError,'passed off'):
            pe.load_data(self.data)
        self.rows[0]=dict(row('x'),is_synthetic=True)
        self.save_data()
        with self.assertRaisesRegex(ValueError,'passed off'):
            pe.load_data(self.data)

    def test_test_review_gate_including_uncertainty(self):
        pe.require_split_reviews(self.rows,'test','human')
        self.rows[2]['label']=None
        pe.require_split_reviews(self.rows,'test','human')
        self.rows[2]['review_status']='pending_human_review'
        with self.assertRaisesRegex(ValueError,'test is closed'):
            pe.require_split_reviews(self.rows,'test','human')

    def test_cache_mismatch_and_integrity(self):
        path=self.cache('cache.json','dev',[.1,.9])
        cache=pe.read_cache(path)
        pe.read_cache(path,cache['header'])
        for field in ('dataset_sha256','config_sha256','split','selection_sha256'):
            altered=dict(cache['header']); altered[field]='changed'
            with self.assertRaisesRegex(ValueError,'Cache mismatch'):
                pe.read_cache(path,altered)
        obj=json.loads(path.read_text()); obj['predictions']['d0']['score']=.99
        path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'integrity'):
            pe.read_cache(path)

    def test_partial_cache_and_aggregation_validation(self):
        path=self.cache('partial.json','dev',[.1])
        cache=pe.read_cache(path)
        pe.validate_predictions(cache,self.rows,complete=False)
        with self.assertRaises(ValueError):
            pe.validate_predictions(cache,self.rows)
        cache['predictions']['d0']['score']=.2
        with self.assertRaisesRegex(ValueError,'aggregation'):
            pe.validate_predictions(cache,self.rows,complete=False)

    def test_dev_config_selection_and_frozen_test(self):
        bad=self.cache('bad.json','dev',[.9,.1])
        good=self.cache('good.json','dev',[.1,.9],{'mode':'window','model':'fake-for-unit-test'})
        selection_path=self.root/'selection.json'
        with contextlib.redirect_stdout(io.StringIO()):
            pe.select(SimpleNamespace(data=self.data,cache=[bad,good],selection=selection_path,label_policy='human'))
        selection=pe.read_sealed(selection_path)
        self.assertEqual(selection['config']['mode'],'window')
        self.assertEqual(selection['threshold'],.9)
        self.assertEqual(len(selection['candidates']),2)
        test=self.cache('test.json','test',[.2,.95],selection['config'],selection)
        output=self.root/'results.json'
        args=SimpleNamespace(data=self.data,cache=test,selection=selection_path,output=output)
        with contextlib.redirect_stdout(io.StringIO()):
            pe.evaluate(args)
        self.assertEqual(pe.read_sealed(output)['metrics']['f1'],1.)
        with self.assertRaisesRegex(ValueError,'dev predictions only'):
            pe.select(SimpleNamespace(data=self.data,cache=[test],selection=self.root/'illegal.json',label_policy='human'))
        with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
            pe.validate_selection(selection,'different dataset',selection['config_sha256'])
        with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
            pe.validate_selection(selection,selection['dataset_sha256'],'different config')


if __name__ == '__main__':
    unittest.main()
