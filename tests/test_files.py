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
            vote_jsonl(RECORDS, voted, final, FIELDS, tiebreaker, prompt, base=3, workers=2, seconds=60)
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

    def test_transport_failure_is_retried_instead_of_aborting(self):
        failures = []

        def flaky(text, seed):
            if not failures:
                failures.append(text)
                raise ConnectionResetError("reset")
            return "age: 20"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voter.jsonl"
            annotate_jsonl(RECORDS[:1], path, FIELDS, flaky, prompt, workers=1, seconds=60, annotators=1, extra_votes=0, retries=1)
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([ballot["error"] is None for ballot in row["ballots"]], [False, True])

    def test_concurrent_voters_file_votes_with_tiebreak_only_when_unresolved(self):
        def collect(directory):
            path = directory / "voters.jsonl"
            annotate_jsonl(RECORDS, path, FIELDS, [voter(answers) for answers in ANSWERS], prompt, workers=2, seconds=60,
                           annotators=3, extra_votes=0)
            return [path]
        self.vote(collect)

    def test_lazy_third_voter_and_tiebreaker_run_in_separate_stages_only_when_unresolved(self):
        third, tie = [], []

        def live(calls, answer):
            return lambda text, seed: calls.append(text) or answer
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = [directory / f"voter{index}.jsonl" for index in range(2)]
            for path, answers in zip(paths, ANSWERS):
                annotate_jsonl(RECORDS, path, FIELDS, voter(answers), prompt, workers=2, seconds=60, annotators=1, extra_votes=0)
            staged, final = directory / "third.jsonl", directory / "final.jsonl"
            vote_jsonl(RECORDS, paths, staged, FIELDS, live(third, "age: 32\ncity: oslo"), prompt, base=2, workers=2, seconds=60)
            vote_jsonl(RECORDS, [staged], final, FIELDS, live(tie, "age: 31\ncity: oslo"), prompt, base=2, workers=2, seconds=60)
            rows = {row["id"]: row for row in map(json.loads, final.read_text(encoding="utf-8").splitlines())}
        self.assertEqual((third, tie), (["two"], ["two"]))
        self.assertEqual(rows["a"]["values"], {"age": "20"})
        self.assertEqual(rows["b"]["values"], {"age": "31", "city": "oslo"})
        self.assertEqual(len(rows["b"]["ballots"]), 4)


if __name__ == "__main__":
    unittest.main()
