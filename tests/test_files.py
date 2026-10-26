import json
import tempfile
import unittest
from pathlib import Path
from local_annotator import Field, Record
from local_annotator.files import annotate_jsonl, tiebreak_jsonl

FIELDS = {"age": Field("Age"), "city": Field("City")}


def prompt(record, fields):
    return record.text


def voter(answers):
    return lambda text, seed: answers[text]


class FileTests(unittest.TestCase):
    def test_tiebreak_runs_only_for_unresolved_records_and_keeps_the_rest(self):
        records = [Record("a", "one"), Record("b", "two")]
        voters = [voter({"one": "age: 20", "two": "age: 30\ncity: oslo"}), voter({"one": "age: 20", "two": "age: 31\ncity: oslo"}),
                  voter({"one": "age: 20", "two": "age: 32\ncity: oslo"})]
        calls = []

        def tiebreaker(text, seed):
            calls.append(text)
            return "age: 31\ncity: oslo"
        with tempfile.TemporaryDirectory() as directory:
            voted, final = Path(directory) / "voted.jsonl", Path(directory) / "final.jsonl"
            annotate_jsonl(records, voted, FIELDS, voters, prompt, workers=2, seconds=60, annotators=3, extra_votes=0)
            tiebreak_jsonl(records, voted, final, FIELDS, tiebreaker, prompt, workers=2, seconds=60)
            rows = {row["id"]: row for row in map(json.loads, final.read_text(encoding="utf-8").splitlines())}
        self.assertEqual(calls, ["two"])
        self.assertEqual(rows["a"]["values"], {"age": "20"})
        self.assertEqual(rows["b"]["values"], {"age": "31", "city": "oslo"})
        self.assertEqual(rows["b"]["unresolved"], [])
        self.assertEqual(len(rows["b"]["ballots"]), 4)


if __name__ == "__main__":
    unittest.main()
