# QLoRA Trainer — AI Jobs Template

Fine-tune any HuggingFace causal LM using QLoRA (4-bit quantized LoRA). Train 7B models on a single T4 or 70B models on an A100.

## How It Works

1. Loads the base model in 4-bit quantization via bitsandbytes (NF4)
2. Applies LoRA adapters to attention projection layers
3. Trains only the LoRA weights (~0.1-1% of total parameters)
4. Saves the adapter weights (typically 10-50MB) — not the full model

## Dataset Formats

### Alpaca Format
```jsonl
{"instruction": "Summarize this article.", "input": "The article text...", "output": "Summary..."}
{"instruction": "Write a haiku about coding.", "input": "", "output": "Bugs hide in the code..."}
```

### ShareGPT Format
```jsonl
{"conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi there!"}]}
```

## Config

```json
{
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "quantization": "4bit",
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.03,
    "max_seq_length": 2048,
    "dataset_format": "alpaca"
}
```

## GPU Requirements

| Model      | GPU  | VRAM Used | Batch Size |
|-----------|------|-----------|------------|
| Gemma 2B   | T4   | ~8GB      | 4-8        |
| Phi-3 3.8B | T4   | ~10GB     | 4          |
| Mistral 7B | T4   | ~12GB     | 2-4        |
| Llama 8B   | T4   | ~14GB     | 2          |
| Llama 70B  | A100 | ~42GB     | 1-2        |

## Output

- `adapter/` — LoRA adapter weights + tokenizer (merge with base model for inference)
- `checkpoints/` — Training checkpoints
- `training_log.jsonl` — Per-step loss and learning rate
- `summary.json` — Training results and config

## Merging the Adapter

After training, merge the adapter back into the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("base-model-id")
model = PeftModel.from_pretrained(base_model, "path/to/adapter")
merged = model.merge_and_unload()
merged.save_pretrained("merged-model")
```
