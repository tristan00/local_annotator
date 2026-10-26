"""User-approved prompts. No additional instructions are prepended or appended."""
from local_annotator import Field

VERSION = "short-prompts-v3"
PROMPTS = {
    "age": '''What age does the author state they are in this comment?
If no age is stated, return null.

Return JSON: {"age": <integer or null>}

Comment:
<COMMENT TEXT>''',
    "appeal": '''What is the main rhetorical appeal of this comment?

- ethos: credibility or authority
- pathos: emotion
- logos: reasoning or evidence

If there is no main appeal, return null.

Return JSON: {"appeal": <"ethos", "pathos", "logos", or null>}

Comment:
<COMMENT TEXT>''',
}
TASKS = {
    "age": {"age": Field("Author age", "integer")},
    "appeal": {"appeal": Field("Main rhetorical appeal", allowed=("ethos", "pathos", "logos"))},
}
SCHEMAS = {
    "age": {"type": "object", "properties": {"age": {"type": ["integer", "null"]}},
            "required": ["age"], "additionalProperties": False},
    "appeal": {"type": "object", "properties": {"appeal": {"type": ["string", "null"],
               "enum": ["ethos", "pathos", "logos", None]}},
               "required": ["appeal"], "additionalProperties": False},
}


def prompt_builder(task):
    def build(record, fields):
        return PROMPTS[task].replace("<COMMENT TEXT>", record.text)
    return build
