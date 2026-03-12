# LM Eval Harness — AI Jobs Template

Runs [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks on any HuggingFace model.

## Supported Benchmarks

- MMLU (57-subject knowledge)
- HellaSwag (commonsense reasoning)
- ARC Easy / Challenge (science reasoning)
- TruthfulQA, Winogrande, GSM8K, and [all lm-eval tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)

## Config

```json
{
    "tasks": ["mmlu", "hellaswag", "arc_easy", "arc_challenge"],
    "num_fewshot": 5,
    "batch_size": "auto",
    "dtype": "float16",
    "max_length": 2048
}
```

## GPU Requirements

| Model Size | GPU   | dtype   | Notes |
|-----------|-------|---------|-------|
| ≤3B       | T4    | float16 | ~6-8GB VRAM |
| 7B        | T4    | float16 | ~14GB VRAM, tight fit |
| 7B        | A100  | float16 | Comfortable |
| 13B+      | A100  | float16 | Requires A100 80GB |

## Output

- `results/` — Per-task JSON results from lm-eval
- `samples/` — Per-sample predictions (with `--log_samples`)
- `summary.json` — AI Jobs summary with scores and timing
