# Full SFT Trainer — AI Jobs Template

Full supervised fine-tuning for small language models (≤3B parameters). Updates all model weights for maximum task-specific quality.

## When to Use

- **Use Full SFT** when: model is small (≤3B), you need maximum quality, you have enough GPU memory
- **Use QLoRA instead** when: model is larger (7B+), GPU memory is limited, or you want faster training

## Dataset Formats

Same as QLoRA trainer — supports Alpaca and ShareGPT formats.

### Alpaca Format
```jsonl
{"instruction": "Summarize this article.", "input": "The article text...", "output": "Summary..."}
```

### ShareGPT Format
```jsonl
{"conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi!"}]}
```

## Config

```json
{
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "fp16": true,
    "save_steps": 200,
    "eval_steps": 200,
    "save_total_limit": 3,
    "dataset_format": "alpaca"
}
```

## GPU Requirements

| Model          | GPU  | VRAM Used | Notes |
|---------------|------|-----------|-------|
| TinyLlama 1.1B | T4   | ~8GB      | Comfortable on T4 |
| SmolLM2 1.7B   | T4   | ~10GB     | Fits on T4 with grad checkpointing |
| Gemma 2B       | T4   | ~12GB     | Tight on T4, comfortable on A100 |
| Phi-3 3.8B     | A100 | ~20GB     | Requires A100 |

Gradient checkpointing is enabled by default to reduce memory usage.

## Output

- `model/` — Full model checkpoint + tokenizer (ready for inference)
- `checkpoints/` — Training checkpoints (best model auto-selected)
- `training_log.jsonl` — Per-step loss and learning rate
- `summary.json` — Training results and config

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/output/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/output/model")

inputs = tokenizer("### Instruction:\nYour prompt here\n\n### Response:\n", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
