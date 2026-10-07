"""Integrity gates for explicitly AI-labeled development review exports."""
import unittest
from build_eval_refresh import validate


class EvalRefreshTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='dev-a', text='Alice survives the final battle.', split='dev',
            group_id='work-a', label=1, label_provenance='ai', review_status='ai_reviewed',
            eligible_for_evaluation=False, reviewer_id='assistant', reviewed_at='2026-10-05T12:00:00+00:00',
            review_rationale='Reveals a later character fate.', revealing_quote='Alice survives', missing_context=None)
        self.original = {'dev-a':dict(self.row)}

    def test_original_text_and_split_are_protected(self):
        for mutation in ({'text':'changed'}, {'split':'test'}, {'group_id':'other'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate([{**self.row, **mutation}], self.original)

    def test_ai_cannot_become_human_or_eligible(self):
        for mutation in ({'label_provenance':'human'}, {'eligible_for_evaluation':True}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate([{**self.row, **mutation}], self.original)

    def test_positive_requires_real_span_and_uncertain_requires_gap(self):
        for mutation in ({'revealing_quote':'Bob dies'}, {'label':None}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate([{**self.row, **mutation}], self.original)

    def test_repeated_text_rejected_even_with_new_id(self):
        with self.assertRaises(ValueError):
            validate([self.row, {**self.row, 'id':'dev-b'}], self.original)
        self.assertEqual(validate([self.row], self.original), {'dev-a'})


if __name__ == '__main__':
    unittest.main()
