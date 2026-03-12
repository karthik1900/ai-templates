# Custom Prompt Eval — AI Jobs Template

Evaluate any HuggingFace model on your own prompt dataset with automated metrics.

## Dataset Format

JSONL file with `prompt` (required) and `reference` (optional) fields:

```jsonl
{"prompt": "What is the capital of France?", "reference": "Paris"}
{"prompt": "Write a Python function to sort a list.", "reference": "def sort_list(lst): return sorted(lst)"}
```

If `reference` is provided, metrics (ROUGE-L, BLEU, exact match) are computed.
If omitted, only completions are generated (useful for manual review).

## Config

```json
{
    "max_new_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "batch_size": 8,
    "dtype": "float16",
    "metrics": ["rouge", "bleu", "exact_match"]
}
```

## GPU Requirements

| Model Size | GPU  | VRAM Usage |
|-----------|------|------------|
| ≤3B       | T4   | ~6-8GB     |
| 7B        | T4   | ~14GB      |
| 13B+      | A100 | ~26GB+     |

## Output

- `predictions.jsonl` — Per-sample prompts, completions, references, and metrics
- `summary.json` — Aggregate metrics and timing
