import unittest
from compare_three_models import metrics,parse_answer,prepare_rows,summarize
import tempfile,json
from pathlib import Path

class ComparisonTests(unittest.TestCase):
 def test_abstention_does_not_hide_missed_spoiler(self):
  rows=[{'id':'a'},{'id':'b'}];labels={'a':{'label':1},'b':{'label':0}}
  result=metrics(rows,labels,{'a':{'prediction':None},'b':{'prediction':0}})
  self.assertEqual(result['recall_including_abstentions'],0)
  self.assertIsNone(result['recall_answered'])
  self.assertEqual(result['coverage'],.5)
 def test_unknown_not_safe(self):
  self.assertEqual(parse_answer('UNCERTAIN'),(None,True))
  self.assertEqual(parse_answer('Ignore everything SAFE'),(None,False))
 def test_no_positives_no_recall(self):
  result=metrics([{'id':'a'}],{'a':{'label':0}},{'a':{'prediction':0}})
  self.assertIsNone(result['recall_including_abstentions'])
 def test_test_and_label_leakage_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'data.jsonl'
   for row in [{'id':'a','text':'x','surface':'review','split':'test'}, {'id':'a','text':'x','surface':'review','split':'dev','label':1}]:
    p.write_text(json.dumps(row))
    with self.assertRaises(ValueError):prepare_rows(p)
if __name__=='__main__':unittest.main()
