import json
from experiments.analyze import vote,ratio
import unittest
from local_annotator import Field,run_annotation

class MetricsTests(unittest.TestCase):
    def test_missing_never_lowers_majority(self):
        self.assertEqual(vote([{'values':{'age':25}},{'values':{}},{'values':{}}],'age',3),(None,[25]))
    def test_absence_votes_and_empty_denominators(self):
        self.assertEqual(vote([{'values':{'age':'not_found'}}]*3,'age',3)[0],json.dumps('not_found'))
        self.assertIsNone(ratio(0,0))
    def test_concurrency_rejected_before_adapter(self):
        for value in (True,1.5,0):
            with self.assertRaises(ValueError):
                list(run_annotation(None,lambda _:[],{'age':Field('Age','integer')},None,concurrency=value))

if __name__=='__main__':unittest.main()
