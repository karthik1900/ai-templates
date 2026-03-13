#!/usr/bin/env python3
"""AI Jobs runner for custom prompt evaluation.

Loads a JSONL dataset with "prompt" and optional "reference" fields,
generates completions from a HuggingFace model, and computes metrics.

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
import math
import os
import sys
import time
from collections import Counter


def load_config() -> dict:
    config_path = os.environ.get("TASK_CONFIG", "/tmp/task_config.json")
    with open(config_path) as f:
        return json.load(f)


def download_s3_dataset(s3_path: str, local_path: str):
    """Download dataset from S3 to local path."""
    import boto3

    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded dataset from s3://{bucket}/{key}")


def load_dataset(dataset_path: str, s3_path: str) -> list[dict]:
    """Load JSONL dataset from local path or S3."""
    if dataset_path and os.path.exists(dataset_path):
        path = dataset_path
    elif s3_path:
        path = "/tmp/dataset.jsonl"
        download_s3_dataset(s3_path, path)
    else:
        print("ERROR: No dataset provided (DATASET_PATH or DATASET_S3_PATH)", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score (no external dependency)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Compute sentence-level BLEU (no external dependency)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
        total = max(sum(pred_ngrams.values()), 1)
        scores.append(clipped / total)

    # Geometric mean
    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))

    return bp * math.exp(log_avg)


def generate_completions(model, tokenizer, prompts: list[str], config: dict) -> list[str]:
    """Generate completions in batches."""
    import torch

    batch_size = config.get("batch_size", 8)
    max_new_tokens = config.get("max_new_tokens", 512)
    temperature = config.get("temperature", 0.0)
    top_p = config.get("top_p", 1.0)

    # Determine generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    completions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = output[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text.strip())

        print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)} completions")

    return completions


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

    # Load dataset
    print(f"Loading dataset...")
    records = load_dataset(dataset_path, s3_path)
    print(f"  Loaded {len(records)} records")

    prompts = [r["prompt"] for r in records]
    references = [r.get("reference", "") for r in records]
    has_references = any(references)

    # Load model
    print(f"\nLoading model: {model_id}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = config.get("dtype", "float16")
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=hf_token or None,
        trust_remote_code=True,
    )
    print(f"  Model loaded on {model.device}")

    # Generate
    print(f"\nGenerating completions...")
    start = time.time()
    completions = generate_completions(model, tokenizer, prompts, config)
    gen_time = time.time() - start
    print(f"  Generation took {gen_time:.1f}s")

    # Compute metrics
    metrics_config = config.get("metrics", ["rouge", "bleu", "exact_match"])
    per_sample = []
    agg_metrics = {}

    if has_references:
        print(f"\nComputing metrics: {metrics_config}")
        all_rouge = []
        all_bleu = []
        all_em = []

        for i, (comp, ref) in enumerate(zip(completions, references)):
            sample = {
                "index": i,
                "prompt": prompts[i],
                "completion": comp,
                "reference": ref,
                "metrics": {},
            }
            if ref:
                if "rouge" in metrics_config:
                    rouge = compute_rouge_l(comp, ref)
                    sample["metrics"]["rouge_l"] = round(rouge, 4)
                    all_rouge.append(rouge)
                if "bleu" in metrics_config:
                    bleu = compute_bleu(comp, ref)
                    sample["metrics"]["bleu"] = round(bleu, 4)
                    all_bleu.append(bleu)
                if "exact_match" in metrics_config:
                    em = compute_exact_match(comp, ref)
                    sample["metrics"]["exact_match"] = em
                    all_em.append(em)
            per_sample.append(sample)

        if all_rouge:
            agg_metrics["rouge_l_avg"] = round(sum(all_rouge) / len(all_rouge), 4)
        if all_bleu:
            agg_metrics["bleu_avg"] = round(sum(all_bleu) / len(all_bleu), 4)
        if all_em:
            agg_metrics["exact_match_avg"] = round(sum(all_em) / len(all_em), 4)
    else:
        for i, comp in enumerate(completions):
            per_sample.append({
                "index": i,
                "prompt": prompts[i],
                "completion": comp,
            })

    elapsed = time.time() - start
    hours = elapsed / 3600

    # Write outputs
    with open(os.path.join(output_dir, "predictions.jsonl"), "w") as f:
        for sample in per_sample:
            f.write(json.dumps(sample) + "\n")

    summary = {
        "task_id": os.environ.get("TASK_ID", ""),
        "model_id": model_id,
        "num_samples": len(records),
        "has_references": has_references,
        "metrics": agg_metrics,
        "generation_config": {
            "max_new_tokens": config.get("max_new_tokens", 512),
            "temperature": config.get("temperature", 0.0),
            "top_p": config.get("top_p", 1.0),
        },
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(hours, 3),
        "throughput_samples_per_sec": round(len(records) / elapsed, 2) if elapsed > 0 else 0,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate charts.json for the UI
    charts = []
    if agg_metrics:
        metric_names = list(agg_metrics.keys())
        metric_values = list(agg_metrics.values())
        display_names = [n.replace("_avg", "").replace("_", " ").upper() for n in metric_names]
        charts.append({
            "data": [{
                "type": "bar",
                "x": display_names,
                "y": metric_values,
                "text": [f"{v:.4f}" for v in metric_values],
                "textposition": "outside",
                "marker": {"color": "#636EFA"},
            }],
            "layout": {
                "title": {"text": f"Average Metrics — {model_id}"},
                "xaxis": {"title": {"text": "Metric"}},
                "yaxis": {"title": {"text": "Score"}, "range": [0, 1]},
                "margin": {"t": 60, "b": 80},
            },
        })
    if has_references and all_rouge:
        charts.append({
            "data": [{
                "type": "histogram",
                "x": [round(v, 4) for v in all_rouge],
                "nbinsx": 20,
                "marker": {"color": "#EF553B"},
            }],
            "layout": {
                "title": {"text": "ROUGE-L Score Distribution"},
                "xaxis": {"title": {"text": "ROUGE-L"}},
                "yaxis": {"title": {"text": "Count"}},
                "margin": {"t": 60, "b": 60},
            },
        })
    if charts:
        with open(os.path.join(output_dir, "charts.json"), "w") as f:
            json.dump(charts, f)

    print(f"\nResults:")
    for k, v in agg_metrics.items():
        print(f"  {k}: {v}")
    print(f"\nCompleted in {elapsed:.0f}s ({hours:.2f} GPU-hours)")
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
