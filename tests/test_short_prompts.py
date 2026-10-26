import unittest
from local_annotator import Record, run_annotation
from experiments.tasks import TASKS, PROMPTS, prompt_builder


class ShortPromptTests(unittest.TestCase):
    def test_age_prompt_matches_approved_text(self):
        expected = ('What age does the author state they are in this comment?\n'
                    'If no age is stated, return null.\n\n'
                    'Return JSON: {"age": <integer or null>}\n\nComment:\nI was 20.')
        self.assertEqual(prompt_builder("age")(Record("a", "I was 20."), TASKS["age"]), expected)

    def test_appeal_prompt_matches_approved_text(self):
        expected = ('What is the main rhetorical appeal of this comment?\n\n'
                    '- ethos: credibility or authority\n- pathos: emotion\n- logos: reasoning or evidence\n\n'
                    'If there is no main appeal, return null.\n\n'
                    'Return JSON: {"appeal": <"ethos", "pathos", "logos", or null>}\n\n'
                    'Comment:\nExample')
        self.assertEqual(prompt_builder("appeal")(Record("a", "Example"), TASKS["appeal"]), expected)

    def run_flat(self, task, replies):
        responses = iter(replies)
        prompts = []
        def model(prompt, seed):
            prompts.append(prompt)
            return next(responses)
        result = next(run_annotation(None, lambda _: [Record("a", "Example")], TASKS[task], model,
            annotators=len(replies), prompt_builder=prompt_builder(task), output_format="flat"))
        self.assertEqual(set(prompts), {PROMPTS[task].replace("<COMMENT TEXT>", "Example")})
        return result

    def test_null_is_valid_absence_but_missing_is_not(self):
        result = self.run_flat("age", ['{"age":null}', '{}', '{"age":null}'])
        self.assertEqual(result.decisions["age"]["answer"], {"status":"not_found"})
        self.assertEqual(result.decisions["age"]["valid_votes"], 2)
        self.assertIn("age", result.ballots[1].invalid)
        self.assertEqual(result.ballots[0].raw, '{"age":null}')

    def test_age_integer_and_appeal_values_resolve(self):
        self.assertEqual(self.run_flat("age", ['{"age":20}'] * 3).decisions["age"]["answer"]["value"], 20)
        self.assertEqual(self.run_flat("appeal", ['{"appeal":"logos"}'] * 3).decisions["appeal"]["answer"]["value"], "logos")

    def test_bad_types_and_unapproved_categories_are_invalid(self):
        for task, response in [("age",'{"age":true}'),("age",'{"age":"20"}'),
                               ("appeal",'{"appeal":"mixed"}')]:
            result=self.run_flat(task,[response])
            self.assertEqual(result.decisions[task]["status"],"unresolved")
