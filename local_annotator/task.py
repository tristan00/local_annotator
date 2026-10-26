import json
import re
from pathlib import Path

from .core import Field, Record, _parse_lines
from .models import MODELS

FREE = r"[a-z0-9][a-z0-9 '.,/&$-]{0,39}"


def literal(text):
    return re.escape(text).replace(r"\ ", " ")


def listing(fields):
    lines, first = [], {}
    for name, field in fields.items():
        line = f"{name}: {field.description}"
        if field.allowed:
            owner = first.setdefault(field.allowed, name)
            line += f" (one of: {' | '.join(field.allowed)})" if owner == name else f" (one of the values listed for {owner})"
        lines.append(line)
    return "\n".join(lines)


class Task:
    def __init__(self, path):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.name, self.template, self.max_lines = spec["name"], spec["prompt"], spec["max_lines"]
        self.voters, self.tiebreaker = spec["voters"], spec["tiebreaker"]
        models = [*self.voters, self.tiebreaker]
        if len(self.voters) != 3 or len(set(models)) != 4:
            raise ValueError(f"need three voters and a tiebreaker, all different models: {models}")
        unknown = [name for name in models if name not in MODELS]
        if unknown:
            raise ValueError(f"unknown models {unknown}")
        self.fields = {name: Field(field["description"], allowed=tuple(field.get("values", ()))) for name, field in spec["fields"].items()}
        patterns = {name: "(" + ("|".join(map(literal, field["values"])) if "values" in field else field.get("pattern", FREE)) + ")"
                    for name, field in spec["fields"].items()}
        line = "(" + "|".join(literal(name) + ": " + pattern for name, pattern in patterns.items()) + ")"
        self.regex = "none|" + line + r"(\n" + line + "){0," + str(self.max_lines - 1) + "}"
        self.messages = [{"role": "system", "content": spec["instructions"].replace("{fields}", listing(self.fields))}]
        compiled = re.compile(self.regex)
        for number, example in enumerate(spec["examples"]):
            if not compiled.fullmatch(example["answer"]):
                raise ValueError(f"example {number} answer does not match the output pattern: {example['answer']!r}")
            _, invalid = _parse_lines(example["answer"], self.fields)
            if invalid:
                raise ValueError(f"example {number} answer is invalid: {invalid}")
            self.messages += [{"role": "user", "content": self.prompt(Record(str(number), example["text"]), self.fields)},
                              {"role": "assistant", "content": example["answer"]}]

    def prompt(self, record, fields):
        return self.template.replace("{text}", record.text)
