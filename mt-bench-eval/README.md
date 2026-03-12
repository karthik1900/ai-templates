# MT-Bench Eval — AI Jobs Template

Multi-turn conversation quality evaluation inspired by [LMSYS MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

Tests 8 categories with 2-turn questions, scored 1-10 by GPT-4 as judge.

## Categories

- Writing, Roleplay, Reasoning, Math, Coding, Extraction, STEM, Humanities

## Config

```json
{
    "num_turns": 2,
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "judge_model": "gpt-4",
    "categories": ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"],
    "dtype": "float16"
}
```

Set `OPENAI_API_KEY` environment variable for GPT-4 judging. Without it, answers are still generated and saved — judgment can be run separately.

## GPU Requirements

Same as other eval templates — model needs to fit in VRAM for inference.

## Output

- `answers.jsonl` — Per-question turns, answers, and judgments
- `summary.json` — Per-category and overall scores
