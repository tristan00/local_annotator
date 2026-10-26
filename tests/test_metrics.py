import unittest
from experiments.report import agreement, panel
from experiments.common import artifact_root
from pathlib import Path


class MetricsTests(unittest.TestCase):
    def test_unresolved_pairs_not_counted_as_agreement(self):
        result = agreement({"a":None,"b":"x","c":"x"},{"a":None,"b":"x","c":None})
        self.assertEqual(result["both_resolved"],1)
        self.assertEqual(result["same_over_all_common"],1/3)

    def test_invalid_ballots_cannot_lower_majority(self):
        ballots=[{"voter":i,"answers":{"age":{"status":"not_found"}} if i<2 else {}} for i in range(5)]
        self.assertIsNone(panel(ballots,"age",5)[0])
        self.assertIsNotNone(panel(ballots,"age",3)[0])

    def test_reject_artifacts_in_source(self):
        with self.assertRaises(ValueError):
            artifact_root(Path(__file__).resolve().parents[1] / "results")
