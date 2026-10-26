# local_annotator

Labels text records with local LLMs served by vLLM inside WSL. Three voter models each label every record, a fourth model votes only on records where the voters split, and any failure stops the run. One command runs everything: serving each model, annotating, voting, tiebreaking and reporting.

## Requirements

- Windows with the WSL distribution `Ubuntu-24.04`, vLLM installed in `/root/vllm`, and model weights in `/root/models/<name>`, where `<name>` is a key of `local_annotator/models.py`.
- A CUDA 13.0 toolkit matching vLLM's FlashInfer JIT, installed once inside WSL with `bash local_annotator/cuda_toolkit.sh`. `local_annotator/vllm_serve.sh` checks it before every server start.
- Python 3.11 or newer on Windows for the client. It uses only the standard library.

## Usage

```
python -m local_annotator TASK RECORDS OUT [--resume]
```

| argument | meaning |
|---|---|
| `TASK` | task file (JSON, below) naming the fields, prompt, examples, three voters and the tiebreaker |
| `RECORDS` | JSONL with one `{"id": ..., "text": ...}` per line; ids must be unique |
| `OUT` | output directory; it must not exist unless `--resume` is given |
| `--resume` | continue a stopped run: finished voter passes are skipped and partial files are continued |

Example, labeling 100,000 Reddit texts:

```
cd D:\local_annotator
python -m local_annotator D:/authorship_id/claims/annotation.json D:/reddit_data/annotations/2026-03/train0/records.jsonl D:/reddit_data/annotations/2026-03/train0/labels
```

The run starts each voter's server in turn, sends every record to it, stops it, then starts the tiebreaker for the records the voters left unresolved. On an RTX 5090 that example took 31.4 minutes, including four server starts.

## Task file

```json
{
 "name": "claims-v1",
 "voters": ["gemma4-26b-a4b-nvfp4", "ministral3-14b", "gpt-oss-20b"],
 "tiebreaker": "qwen3.6-35b-a3b-nvfp4",
 "prompt": "Text:\n{text}",
 "max_lines": 12,
 "instructions": "You label what Reddit authors say about themselves. ...\n\nFields:\n{fields}",
 "fields": {
  "age": {"description": "age in years", "pattern": "1[3-9]|[2-9][0-9]|100"},
  "gender": {"description": "gender identity", "values": ["female", "male", "non-binary"]},
  "residence_city": {"description": "city or town they live in", "pattern": "[a-z][a-z .'-]{0,39}"}
 },
 "examples": [
  {"text": "34F here.", "answer": "age: 34\ngender: female"},
  {"text": "What a game last night.", "answer": "none"}
 ]
}
```

| key | meaning |
|---|---|
| `voters` | three different model names from `models.py`; each labels every record |
| `tiebreaker` | a fourth, different model; it only sees records with an unresolved field |
| `prompt` | the user message for one record; `{text}` is replaced by the record text |
| `instructions` | the system message; `{fields}` is replaced by one line per field with its description and allowed values |
| `max_lines` | the most answer lines a model may write |
| `fields` | per field, a `description` and either `values` (a closed list) or `pattern` (a regular expression); without either, a short free-text pattern is used |
| `examples` | few-shot turns sent before every record; each answer must match the output format |

Answers are `none` or lines of `field: value`. Decoding is constrained by a regular expression built from the fields, so every answer is well formed and every closed-list value is allowed.

## Output

| file | content |
|---|---|
| `OUT/voters/<model>.jsonl` | each voter's raw answer per record |
| `OUT/voters/<model>.timing.json` | that pass's startup, annotation and stop seconds and its rate |
| `OUT/labels.jsonl` | per record: `values` (resolved field values), `unresolved` (fields with no majority) and every ballot |
| `OUT/report.json` | passes, tiebreak calls, invalid answers per voter, pairwise voter agreement per field, unresolved and resolved counts, total seconds |

## Voting

A field resolves when one answer, a value or `none`, has at least two votes and more votes than any other answer. A record with any unresolved field gets the tiebreaker's vote; fields that already resolved keep their result. An invalid answer abstains for that field. Transport errors are retried once and then stop the run, so no record is silently dropped.

## Settings

Constants at the top of `local_annotator/run.py`:

| constant | default | meaning |
|---|---|---|
| `PORT` | 8120 | local port of the vLLM server |
| `WORKERS` | 768 | concurrent client requests |
| `GPU` | 0.88 | vLLM GPU memory utilization |
| `MAX_MODEL_LEN` | 16384 | longest prompt plus answer, in tokens |
| `STARTUP_SECONDS` | 1800 | how long to wait for a server to become healthy |

`local_annotator/models.py` is the model registry. Each entry gives the Hugging Face repository, extra `vllm serve` arguments, sampling settings and the answer token limit. To add a model, download it to `/root/models/<name>` and add an entry.

A model can also be served by hand, for example to test a new registry entry:

```
python -m local_annotator.serve start NAME PORT GPU [MAX_NUM_SEQS MAX_MODEL_LEN MINUTES STARTUP_SECONDS]
python -m local_annotator.serve stop NAME
```

## Tests

```
python -m unittest discover -s tests
```
