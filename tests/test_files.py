import json
import tempfile
import unittest
from pathlib import Path
from local_annotator import Field, Record
from local_annotator.files import annotate_jsonl, vote_jsonl

FIELDS = {"age": Field("Age"), "city": Field("City")}
RECORDS = [Record("a", "one"), Record("b", "two")]
ANSWERS = [{"one": "age: 20", "two": "age: 30\ncity: oslo"}, {"one": "age: 20", "two": "age: 31\ncity: oslo"},
           {"one": "age: 20", "two": "age: 32\ncity: oslo"}]


def prompt(record, fields):
    return record.text


def voter(answers):
    return lambda text, seed: answers[text]


class FileTests(unittest.TestCase):
    def vote(self, collect):
        calls = []

        def tiebreaker(text, seed):
            calls.append(text)
            return "age: 31\ncity: oslo"
        with tempfile.TemporaryDirectory() as directory:
            voted = collect(Path(directory))
            final = Path(directory) / "final.jsonl"
            vote_jsonl(RECORDS, voted, final, FIELDS, tiebreaker, prompt, workers=2, seconds=60)
            rows = {row["id"]: row for row in map(json.loads, final.read_text(encoding="utf-8").splitlines())}
        self.assertEqual(calls, ["two"])
        self.assertEqual(rows["a"]["values"], {"age": "20"})
        self.assertEqual(rows["b"]["values"], {"age": "31", "city": "oslo"})
        self.assertEqual(rows["b"]["unresolved"], [])
        self.assertEqual(len(rows["b"]["ballots"]), 4)

    def test_staged_single_model_passes_vote_with_tiebreak_only_when_unresolved(self):
        def collect(directory):
            paths = [directory / f"voter{index}.jsonl" for index in range(3)]
            for path, answers in zip(paths, ANSWERS):
                annotate_jsonl(RECORDS, path, FIELDS, voter(answers), prompt, workers=2, seconds=60, annotators=1, extra_votes=0)
            return paths
        self.vote(collect)

    def test_concurrent_voters_file_votes_with_tiebreak_only_when_unresolved(self):
        def collect(directory):
            path = directory / "voters.jsonl"
            annotate_jsonl(RECORDS, path, FIELDS, [voter(answers) for answers in ANSWERS], prompt, workers=2, seconds=60,
                           annotators=3, extra_votes=0)
            return [path]
        self.vote(collect)


if __name__ == "__main__":
    unittest.main()
