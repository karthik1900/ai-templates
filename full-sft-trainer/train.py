#!/usr/bin/env python3
"""AI Jobs runner for full supervised fine-tuning (SFT).

Updates all model weights. Best for small models (≤3B) where you want
maximum quality on a specific task.

Environment variables (set by AI Jobs executor):
    TASK_ID          - Unique task identifier
    TASK_CONFIG      - Path to JSON config file
    TASK_OUTPUT_DIR  - Directory to write results
    MODEL_ID         - HuggingFace model ID
    DATASET_PATH     - Path to JSONL dataset
    DATASET_S3_PATH  - S3 URI for dataset (if not local)
    HF_TOKEN         - (optional) HuggingFace auth token
"""

import json
import os
import sys
import time


def load_config() -> dict:
    config_path = os.environ.get("TASK_CONFIG", "/tmp/task_config.json")
    with open(config_path) as f:
        return json.load(f)


def download_s3_dataset(s3_path: str, local_path: str):
    """Download dataset from S3."""
    import boto3
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded dataset from s3://{bucket}/{key}")


def load_dataset_records(dataset_path: str, s3_path: str) -> list[dict]:
    """Load JSONL dataset from local path or S3."""
    if dataset_path and os.path.exists(dataset_path):
        path = dataset_path
    elif s3_path:
        path = "/tmp/dataset.jsonl"
        download_s3_dataset(s3_path, path)
    else:
        print("ERROR: No dataset provided", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_alpaca(record: dict) -> str:
    """Format an Alpaca-style record into a training prompt."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    output = record.get("output", "")

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def format_sharegpt(record: dict, tokenizer) -> str:
    """Format a ShareGPT-style record using the model's chat template."""
    conversations = record.get("conversations", [])
    messages = []
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        text = ""
        for msg in messages:
            text += f"<|{msg['role']}|>\n{msg['content']}\n"
        return text


def main():
    config = load_config()
    model_id = os.environ.get("MODEL_ID", "")
    output_dir = os.environ.get("TASK_OUTPUT_DIR", "/output")
    dataset_path = os.environ.get("DATASET_PATH", "")
    s3_path = os.environ.get("DATASET_S3_PATH", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not model_id:
        print("ERROR: MODEL_ID is required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Config
    learning_rate = config.get("learning_rate", 5e-5)
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("per_device_train_batch_size", 2)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    warmup_ratio = config.get("warmup_ratio", 0.05)
    weight_decay = config.get("weight_decay", 0.01)
    max_seq_length = config.get("max_seq_length", 2048)
    use_fp16 = config.get("fp16", True)
    logging_steps = config.get("logging_steps", 10)
    save_steps = config.get("save_steps", 200)
    eval_steps = config.get("eval_steps", 200)
    save_total_limit = config.get("save_total_limit", 3)
    dataset_format = config.get("dataset_format", "alpaca")

    print(f"Full Supervised Fine-tuning")
    print(f"  Model:          {model_id}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch size:     {batch_size} x {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"  Weight decay:   {weight_decay}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  FP16:           {use_fp16}")
    print(f"  Dataset format: {dataset_format}")
    print()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        TrainerCallback,
    )
    from datasets import Dataset

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token or None, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (full precision, no quantization)
    print("Loading model (full weights)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map="auto",
        token=hf_token or None,
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} (all trainable)")

    # Warn if model is too large for full SFT
    param_billions = total_params / 1e9
    if param_billions > 3.5:
        print(f"\nWARNING: Model has {param_billions:.1f}B parameters.")
        print("Full SFT on models >3B may OOM on A100 40GB. Consider using QLoRA instead.")

    # Load and format dataset
    print(f"\nLoading dataset ({dataset_format} format)...")
    records = load_dataset_records(dataset_path, s3_path)
    print(f"  Loaded {len(records)} records")

    if dataset_format == "sharegpt":
        texts = [format_sharegpt(r, tokenizer) for r in records]
    else:
        texts = [format_alpaca(r) for r in records]

    # Tokenize
    def tokenize(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = split["test"].map(tokenize, batched=True, remove_columns=["text"])

    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Metrics callback
    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self.logs = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 2)}
                for k in ["loss", "learning_rate", "eval_loss"]:
                    if k in logs:
                        entry[k] = round(logs[k], 6) if isinstance(logs[k], float) else logs[k]
                self.logs.append(entry)
                if "loss" in entry:
                    print(f"  Step {entry['step']:5d} | Loss: {entry.get('loss', 'N/A'):8.4f} | LR: {entry.get('learning_rate', 'N/A')}")

    metrics_cb = MetricsCallback()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=use_fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,  # saves memory for full SFT
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[metrics_cb],
    )

    # Train
    print(f"\nStarting training (gradient checkpointing enabled)...")
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start
    hours = elapsed / 3600

    # Save final model
    model_dir = os.path.join(output_dir, "model")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\nModel saved to {model_dir}")

    # Final eval
    eval_result = trainer.evaluate()

    # Write training log
    with open(os.path.join(output_dir, "training_log.jsonl"), "w") as f:
        for entry in metrics_cb.logs:
            f.write(json.dumps(entry) + "\n")

    # Write summary
    summary = {
        "task_id": os.environ.get("TASK_ID", ""),
        "model_id": model_id,
        "method": "full_sft",
        "training_config": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "max_seq_length": max_seq_length,
            "weight_decay": weight_decay,
            "fp16": use_fp16,
            "gradient_checkpointing": True,
        },
        "dataset": {
            "format": dataset_format,
            "total_records": len(records),
            "train_records": len(train_dataset),
            "eval_records": len(eval_dataset),
        },
        "model_params": {
            "total": total_params,
            "total_billions": round(param_billions, 2),
        },
        "results": {
            "train_loss": round(train_result.training_loss, 4),
            "eval_loss": round(eval_result.get("eval_loss", 0), 4),
            "train_steps": train_result.global_step,
        },
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(hours, 3),
        "artifacts": {
            "model": "model/",
            "training_log": "training_log.jsonl",
            "checkpoints": "checkpoints/",
        },
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training Complete")
    print(f"{'='*50}")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Eval loss:  {eval_result.get('eval_loss', 'N/A')}")
    print(f"  Steps:      {train_result.global_step}")
    print(f"  Time:       {elapsed:.0f}s ({hours:.2f} GPU-hours)")
    print(f"  Model:      {model_dir}")


if __name__ == "__main__":
    main()
