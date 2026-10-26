import json
import unittest
from local_annotator import Field, Record, run_annotation


def answer(value):
    return json.dumps({"age": {"status": "value", "value": value}})


class AnnotationTests(unittest.TestCase):
    def run_votes(self, responses, **kwargs):
        replies = iter(responses)
        prompts = []
        def model(prompt, seed):
            prompts.append(prompt)
            value = next(replies)
            if isinstance(value, Exception):
                raise value
            return value
        results = list(run_annotation(None, lambda _: [Record("a", "I am 25")],
                                     {"age": Field("Explicit current age", "integer")},
                                     model, **kwargs))
        self.assertEqual(len(set(prompts)), 1)
        return results[0]

    def test_absence_is_distinct_from_invalid_and_transport_failure(self):
        result = self.run_votes(['{"age":{"status":"not_found"}}', '{}', TimeoutError()])
        self.assertEqual(result.decisions["age"]["status"], "unresolved")
        self.assertEqual(result.decisions["age"]["valid_votes"], 1)
        self.assertIn("age", result.ballots[1].invalid)
        self.assertIn("TimeoutError", result.ballots[2].error)

    def test_single_valid_vote_never_resolves(self):
        result = self.run_votes([answer(25), '{}', '{}'])
        self.assertEqual(result.decisions["age"]["status"], "unresolved")
        self.assertEqual(result.decisions["age"]["scheduled_voters"], 3)

    def test_tied_largest_share_is_unresolved(self):
        result = self.run_votes([answer(25), answer(25), answer(26), answer(26)], annotators=4)
        self.assertEqual(result.decisions["age"]["status"], "unresolved")

    def test_unique_largest_share_resolves_after_extra_vote(self):
        result = self.run_votes([answer(20), answer(21), answer(22), answer(21)],
                                extra_votes=3, disagreement=lambda _: True)
        self.assertEqual(len(result.ballots), 4)
        self.assertEqual(result.decisions["age"]["answer"]["value"], 21)

    def test_each_voter_slot_uses_its_own_model(self):
        calls = []
        def voter(value):
            def model(prompt, seed):
                calls.append(value)
                return answer(value)
            return model
        result = next(run_annotation(None, lambda _: [Record("a", "I am 25")],
                                     {"age": Field("Explicit current age", "integer")},
                                     [voter(20), voter(21), voter(22), voter(21)], annotators=3,
                                     extra_votes=1, disagreement=lambda _: True))
        self.assertEqual(calls, [20, 21, 22, 21])
        self.assertEqual(result.decisions["age"]["answer"]["value"], 21)
        with self.assertRaises(ValueError):
            next(run_annotation(None, lambda _: [Record("a", "x")], {"age": Field("Age", "integer")},
                                [voter(1), voter(2)], annotators=3))

    def test_retries_do_not_create_extra_votes(self):
        result = self.run_votes([TimeoutError(), answer(25), answer(26), answer(26)], retries=1)
        self.assertEqual(len(result.ballots), 4)
        self.assertEqual(result.decisions["age"]["answer"]["value"], 26)
        self.assertEqual(result.decisions["age"]["valid_votes"], 3)

    def test_extra_votes_are_bounded(self):
        result = self.run_votes([answer(20), answer(21), answer(22), answer(23), answer(24)],
                                extra_votes=2, disagreement=lambda _: True)
        self.assertEqual(len(result.ballots), 5)
        self.assertEqual(result.decisions["age"]["status"], "unresolved")

    def test_majority_does_not_request_extra_work(self):
        result = self.run_votes([answer(25)] * 3, extra_votes=5, disagreement=lambda _: True)
        self.assertEqual(len(result.ballots), 3)

    def test_invalid_types_duplicate_keys_and_nonfinite_values(self):
        for raw in [answer(True), answer(25.0), answer(None), answer("25"),
                    '{"age":{"status":"value","value":25,"value":26}}',
                    '{"age":{"status":"value","value":NaN}}']:
            with self.subTest(raw=raw):
                result = self.run_votes([raw], annotators=1)
                self.assertEqual(result.decisions["age"]["valid_votes"], 0)

    def test_persistence_failure_propagates(self):
        def fail(ballot):
            raise OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            self.run_votes([answer(25)], annotators=1, on_ballot=fail)

    def test_observer_cannot_change_the_votes(self):
        def mutate(ballot):
            ballot.answers.clear()
        result = self.run_votes([answer(25)] * 3, on_ballot=mutate)
        self.assertEqual(result.decisions["age"]["answer"]["value"], 25)

    def test_duplicate_record_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate record"):
            list(run_annotation(None, lambda _: [Record("a", "x")] * 2,
                                {"age": Field("age", "integer")}, lambda *_: answer(25)))

    def test_resolution_is_frozen_per_field(self):
        replies = iter([json.dumps({"a": {"status":"value", "value": a},
                                    "b": {"status":"value", "value": b}})
                        for a, b in [(1,1),(1,2),(2,3),(2,2),(2,2)]])
        result = next(run_annotation(None, lambda _: [Record("a", "text")],
            {"a": Field("a", "integer"), "b": Field("b", "integer")},
            lambda *_: next(replies), extra_votes=2, disagreement=lambda _: True))
        self.assertEqual(result.decisions["a"]["answer"]["value"], 1)
        self.assertEqual(result.decisions["a"]["scheduled_voters"], 3)
        self.assertEqual(result.decisions["b"]["answer"]["value"], 2)
        self.assertEqual(result.decisions["b"]["scheduled_voters"], 4)


if __name__ == "__main__":
    unittest.main()
