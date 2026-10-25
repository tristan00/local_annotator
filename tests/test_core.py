import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from local_annotator import AttemptStore, Field, ModelResponse, run_annotation


class Sequence:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.messages = []

    def __call__(self, messages, **kwargs):
        self.messages.append(messages)
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer


class VotingTests(unittest.TestCase):
    def run_case(self, answers, **kwargs):
        adapter = Sequence(answers)
        fields = kwargs.pop("fields", {"age": Field("Explicit current author age", "integer")})
        result = next(run_annotation(None, lambda _: [{"id": "x", "text": "I am 25."}],
                                     fields, adapter, **kwargs))
        return result, adapter

    def test_majority_and_isolation(self):
        result, adapter = self.run_case(['{"age":25}', '{"age":26}', '{"age":25}'])
        self.assertEqual(result["fields"]["age"]["value"], 25)
        self.assertEqual(adapter.messages[0], adapter.messages[1])
        self.assertIsNot(adapter.messages[0], adapter.messages[1])

    def test_absence_is_a_valid_vote(self):
        result, _ = self.run_case(['{"age":"not_found"}'] * 2 + ['{}'])
        self.assertEqual(result["fields"]["age"]["value"], "not_found")

    def test_failures_do_not_lower_threshold(self):
        result, _ = self.run_case(['{"age":25}', TimeoutError(), '{}'])
        self.assertEqual(result["fields"]["age"]["status"], "unresolved")
        self.assertEqual(result["ballots"][1]["attempts"][0]["status"], "failed")

    def test_even_count_requires_strict_majority(self):
        result, _ = self.run_case(['{"age":25}'] * 2 + ['{"age":26}'] * 2, annotator_count=4)
        self.assertEqual(result["fields"]["age"]["threshold"], 3)
        self.assertEqual(result["fields"]["age"]["status"], "unresolved")

    def test_no_bool_as_integer_or_implicit_numeric_string(self):
        result, _ = self.run_case(['{"age":true}', '{"age":"25"}', '{"age":25}'])
        self.assertEqual(result["fields"]["age"]["status"], "unresolved")

    def test_declared_normalization(self):
        result, _ = self.run_case(['{"age":"25"}', '{"age":25}', '{}'],
                                  fields={"age": Field("Age", "integer", integer_strings=True)})
        self.assertEqual(result["fields"]["age"]["value"], 25)

    def test_retries_preserve_valid_fields(self):
        fields = {"age": Field("Age", "integer"), "appeal": Field("Appeal", allowed=("logos", "ethos"))}
        result, _ = self.run_case(['{"age":25}', '{"age":26,"appeal":"logos"}'],
                                  fields=fields, annotator_count=1, retries=1)
        self.assertEqual(result["fields"]["age"]["value"], 25)
        self.assertEqual(result["fields"]["appeal"]["value"], "logos")
        self.assertEqual(len(result["ballots"][0]["attempts"]), 2)

    def test_extra_votes_bounded(self):
        result, _ = self.run_case(['{"age":25}', '{"age":26}', '{"age":27}', '{"age":25}', '{"age":25}'],
                                  disagreement="extra", extra_votes=2)
        self.assertEqual(result["fields"]["age"]["value"], 25)
        self.assertEqual(result["fields"]["age"]["threshold"], 3)
        self.assertEqual(len(result["ballots"]), 5)

    def test_extra_votes_freeze_initial_field_winner(self):
        fields = {"age": Field("Age", "integer"), "appeal": Field("Appeal")}
        result, _ = self.run_case([
            '{"age":25,"appeal":"a"}', '{"age":25,"appeal":"b"}', '{"age":26,"appeal":"c"}',
            '{"age":26,"appeal":"a"}', '{"age":26,"appeal":"a"}'],
            fields=fields, disagreement="extra", extra_votes=2)
        self.assertEqual(result["fields"]["age"]["value"], 25)
        self.assertEqual(result["fields"]["age"]["method"], "initial_majority")
        self.assertEqual(result["fields"]["appeal"]["value"], "a")
        self.assertEqual(result["fields"]["appeal"]["method"], "extra_majority")

    def test_callback_provenance_and_error(self):
        result, _ = self.run_case(['{}'] * 3, disagreement=lambda *args: 31)
        self.assertEqual(result["fields"]["age"]["method"], "callback")
        result, _ = self.run_case(['{}'] * 3, disagreement=lambda *args: "wrong")
        self.assertIn("callback_error", result["fields"]["age"])

    def test_duplicate_keys_and_truncation_invalid(self):
        for raw in ['{"age":25,"age":26}', ModelResponse('{"age":25}', {"finish_reason":"length"})]:
            result, _ = self.run_case([raw] * 3)
            self.assertEqual(result["fields"]["age"]["status"], "unresolved")

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            list(run_annotation(None, lambda _: [{"id": "x", "text": "a"}] * 2,
                                {"age": Field("Age", "integer")}, Sequence(['{"age":25}']), annotator_count=1))

    def test_attempts_durable_before_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.db"
            with AttemptStore(path) as store:
                run = store.start_run({"test": True})
                result, _ = self.run_case(['{}', '{"age":25}'], annotator_count=1, retries=1,
                                          on_attempt=store.attempt_sink(run))
                with closing(sqlite3.connect(path)) as reader:
                    self.assertEqual(reader.execute("SELECT count(*) FROM attempts").fetchone()[0], 2)
                    self.assertEqual(reader.execute("SELECT count(*) FROM results").fetchone()[0], 0)
                store.save_result(run, result)
            with closing(sqlite3.connect(path)) as reader:
                self.assertEqual(json.loads(reader.execute("SELECT payload FROM results").fetchone()[0])["id"], "x")


if __name__ == "__main__":
    unittest.main()
