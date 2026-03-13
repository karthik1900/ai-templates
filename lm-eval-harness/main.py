#!/usr/bin/env python3
"""AI Jobs runner for EleutherAI's lm-evaluation-harness.

Environment variables (set by AI Jobs executor):
    TASK_ID          - Unique task identifier
    TASK_CONFIG      - Path to JSON config file
    TASK_OUTPUT_DIR  - Directory to write results
    MODEL_ID         - HuggingFace model ID
    HF_TOKEN         - (optional) HuggingFace auth token
    DATASET_PATH     - (optional) path to custom dataset
"""

import json
import os
import subprocess
import sys
import time


def load_config() -> dict:
    config_path = os.environ.get("TASK_CONFIG", "/tmp/task_config.json")
    with open(config_path) as f:
        return json.load(f)


def main():
    config = load_config()
    model_id = os.environ.get("MODEL_ID", "")
    output_dir = os.environ.get("TASK_OUTPUT_DIR", "/output")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not model_id:
        print("ERROR: MODEL_ID is required", file=sys.stderr)
        sys.exit(1)

    # Extract config values with defaults
    tasks = config.get("tasks", ["mmlu", "hellaswag"])
    num_fewshot = config.get("num_fewshot", 5)
    batch_size = config.get("batch_size", "auto")
    dtype = config.get("dtype", "float16")
    max_length = config.get("max_length", 2048)

    if isinstance(tasks, list):
        tasks_str = ",".join(tasks)
    else:
        tasks_str = tasks

    # Build model_args
    model_args = f"pretrained={model_id},dtype={dtype}"
    if max_length:
        model_args += f",max_length={max_length}"
    if hf_token:
        model_args += f",token={hf_token}"

    # Build lm_eval command
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", output_dir,
        "--log_samples",
    ]

    print(f"Running lm-evaluation-harness")
    print(f"  Model:      {model_id}")
    print(f"  Tasks:      {tasks_str}")
    print(f"  Few-shot:   {num_fewshot}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dtype:      {dtype}")
    print(f"  Output:     {output_dir}")
    print()

    start = time.time()

    result = subprocess.run(cmd, text=True)

    elapsed = time.time() - start
    hours = elapsed / 3600

    # Write a summary file
    summary = {
        "task_id": os.environ.get("TASK_ID", ""),
        "model_id": model_id,
        "tasks": tasks_str.split(","),
        "num_fewshot": num_fewshot,
        "dtype": dtype,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(hours, 3),
        "exit_code": result.returncode,
    }

    # Try to read the lm_eval results and include scores in summary
    results_dir = os.path.join(output_dir, "results")
    if os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith(".json"):
                with open(os.path.join(results_dir, fname)) as f:
                    eval_results = json.load(f)
                if "results" in eval_results:
                    summary["scores"] = {}
                    for task_name, task_results in eval_results["results"].items():
                        # Extract the primary metric (usually acc or acc_norm)
                        for metric in ["acc_norm,none", "acc,none", "exact_match,none"]:
                            if metric in task_results:
                                summary["scores"][task_name] = {
                                    "metric": metric.split(",")[0],
                                    "value": task_results[metric],
                                }
                                break
                break

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate charts.json for the UI
    charts = []
    if summary.get("scores"):
        task_names = list(summary["scores"].keys())
        values = [summary["scores"][t]["value"] for t in task_names]
        metrics = [summary["scores"][t]["metric"] for t in task_names]
        charts.append({
            "data": [{
                "type": "bar",
                "x": task_names,
                "y": values,
                "text": [f"{v:.3f} ({m})" for v, m in zip(values, metrics)],
                "textposition": "outside",
                "marker": {"color": "#636EFA"},
            }],
            "layout": {
                "title": {"text": f"Benchmark Scores — {model_id}"},
                "xaxis": {"title": {"text": "Task"}},
                "yaxis": {"title": {"text": "Score"}, "range": [0, 1]},
                "margin": {"t": 60, "b": 80},
            },
        })
    if charts:
        with open(os.path.join(output_dir, "charts.json"), "w") as f:
            json.dump(charts, f)

    print(f"\nCompleted in {elapsed:.0f}s ({hours:.2f} GPU-hours)")

    if result.returncode != 0:
        print(f"lm_eval exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
