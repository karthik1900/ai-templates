#!/usr/bin/env python3
"""AI Jobs runner for MT-Bench evaluation.

Generates model answers on MT-Bench's 80 multi-turn questions and
optionally judges them using a GPT-4 judge.

Environment variables (set by AI Jobs executor):
    TASK_ID          - Unique task identifier
    TASK_CONFIG      - Path to JSON config file
    TASK_OUTPUT_DIR  - Directory to write results
    MODEL_ID         - HuggingFace model ID
    HF_TOKEN         - (optional) HuggingFace auth token
    OPENAI_API_KEY   - (optional) for GPT-4 judge
"""

import json
import os
import sys
import time

# MT-Bench categories and sample questions (subset for demo)
MT_BENCH_QUESTIONS = [
    {"question_id": 1, "category": "writing", "turns": [
        "Write a persuasive email to convince your introverted friend to attend a large social event.",
        "Now make the email more concise and add a postscript that includes a specific incentive for attending."
    ]},
    {"question_id": 2, "category": "writing", "turns": [
        "Write a short story about a detective solving a mystery in a small town.",
        "Rewrite the story from the perspective of the culprit."
    ]},
    {"question_id": 3, "category": "roleplay", "turns": [
        "Pretend you are a medieval knight. Describe your typical day.",
        "Now assume you've been transported to modern times. How would you react to a smartphone?"
    ]},
    {"question_id": 4, "category": "roleplay", "turns": [
        "You are a Martian ambassador visiting Earth for the first time. Describe your impressions of a supermarket.",
        "Based on your supermarket visit, what would you report back to Mars about human civilization?"
    ]},
    {"question_id": 5, "category": "reasoning", "turns": [
        "A farmer has 17 sheep and all but 9 die. How many are left?",
        "If the farmer then buys 3 more sheep and 2 of the total sheep run away, how many does the farmer have?"
    ]},
    {"question_id": 6, "category": "reasoning", "turns": [
        "How many times does the letter 'e' appear in the sentence: 'The quick brown fox jumps over the lazy dog'?",
        "Now count how many words in that sentence have exactly 3 letters."
    ]},
    {"question_id": 7, "category": "math", "turns": [
        "What is the sum of all prime numbers less than 20?",
        "Can you verify your answer by listing each prime number and showing the addition step by step?"
    ]},
    {"question_id": 8, "category": "math", "turns": [
        "A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves Station B (300 miles away) at 10:00 AM traveling toward Station A at 80 mph. When do they meet?",
        "If the first train had a 30-minute delay and left at 9:30 AM instead, when would they meet?"
    ]},
    {"question_id": 9, "category": "coding", "turns": [
        "Write a Python function that finds the longest palindromic substring in a given string.",
        "Now optimize your solution to have O(n) time complexity using Manacher's algorithm."
    ]},
    {"question_id": 10, "category": "coding", "turns": [
        "Implement a basic LRU cache in Python with get and put operations.",
        "Now add a method to the cache that returns the most frequently accessed keys."
    ]},
    {"question_id": 11, "category": "extraction", "turns": [
        "Extract all the key facts from this paragraph: 'Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined in 2004 as chairman and led the Series A funding. The company's first car, the Roadster, was launched in 2008 and could travel 245 miles on a single charge.'",
        "Now organize those facts into a structured JSON format."
    ]},
    {"question_id": 12, "category": "extraction", "turns": [
        "Summarize the following: Machine learning models can be categorized into supervised learning (classification and regression), unsupervised learning (clustering and dimensionality reduction), and reinforcement learning. Each paradigm has distinct use cases and requirements for training data.",
        "Create a comparison table of these three paradigms with columns: Type, Subtypes, Data Requirements, Example Use Case."
    ]},
    {"question_id": 13, "category": "stem", "turns": [
        "Explain how mRNA vaccines work in simple terms that a high school student could understand.",
        "What are the key differences between mRNA vaccines and traditional vaccines?"
    ]},
    {"question_id": 14, "category": "stem", "turns": [
        "Explain the concept of entropy in thermodynamics.",
        "How does the concept of entropy relate to information theory?"
    ]},
    {"question_id": 15, "category": "humanities", "turns": [
        "What were the main causes of the French Revolution?",
        "How do those causes compare to the factors leading to the American Revolution?"
    ]},
    {"question_id": 16, "category": "humanities", "turns": [
        "Explain the philosophical concept of the 'Ship of Theseus'.",
        "How does this thought experiment relate to modern debates about personal identity and consciousness?"
    ]},
]


def load_config() -> dict:
    config_path = os.environ.get("TASK_CONFIG", "/tmp/task_config.json")
    with open(config_path) as f:
        return json.load(f)


def generate_answer(model, tokenizer, conversation: list[dict], config: dict) -> str:
    """Generate a response given a conversation history."""
    import torch

    max_new_tokens = config.get("max_new_tokens", 1024)
    temperature = config.get("temperature", 0.0)

    # Format as chat
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def judge_answer(question: dict, answers: list[str], judge_model: str = "gpt-4") -> dict:
    """Use OpenAI API to judge the quality of answers."""
    try:
        from openai import OpenAI
    except ImportError:
        return {"score": -1, "comment": "OpenAI package not installed"}

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"score": -1, "comment": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    prompt = f"""Please act as an impartial judge and evaluate the quality of the responses provided by an AI assistant to the user questions below. Rate the overall quality on a scale of 1 to 10, where 1 is the worst and 10 is the best.

Category: {question['category']}

[Question Turn 1]
{question['turns'][0]}

[Assistant Turn 1]
{answers[0]}
"""
    if len(question["turns"]) > 1 and len(answers) > 1:
        prompt += f"""
[Question Turn 2]
{question['turns'][1]}

[Assistant Turn 2]
{answers[1]}
"""

    prompt += """
Evaluate the response(s) for helpfulness, relevance, accuracy, depth, creativity, and level of detail.

Output your rating in the following JSON format:
{"score": <integer 1-10>, "comment": "<brief explanation>"}
"""

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        result_text = response.choices[0].message.content.strip()
        # Extract JSON from response
        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(result_text[start:end])
        return {"score": -1, "comment": f"Could not parse judge response: {result_text}"}
    except Exception as e:
        return {"score": -1, "comment": f"Judge API error: {str(e)}"}


def main():
    config = load_config()
    model_id = os.environ.get("MODEL_ID", "")
    output_dir = os.environ.get("TASK_OUTPUT_DIR", "/output")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not model_id:
        print("ERROR: MODEL_ID is required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    num_turns = config.get("num_turns", 2)
    judge_model = config.get("judge_model", "gpt-4")
    categories = config.get("categories", [])
    dtype = config.get("dtype", "float16")

    # Filter questions by category if specified
    questions = MT_BENCH_QUESTIONS
    if categories:
        questions = [q for q in questions if q["category"] in categories]

    print(f"MT-Bench Evaluation")
    print(f"  Model:      {model_id}")
    print(f"  Questions:  {len(questions)}")
    print(f"  Turns:      {num_turns}")
    print(f"  Judge:      {judge_model}")
    print(f"  Categories: {categories or 'all'}")
    print()

    # Load model
    print(f"Loading model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token or None, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=hf_token or None,
        trust_remote_code=True,
    )
    print(f"  Model loaded on {model.device}")

    # Generate answers
    start = time.time()
    results = []

    for qi, question in enumerate(questions):
        print(f"\nQuestion {qi + 1}/{len(questions)} [{question['category']}]")
        conversation = []
        answers = []

        for turn_idx in range(min(num_turns, len(question["turns"]))):
            user_msg = question["turns"][turn_idx]
            conversation.append({"role": "user", "content": user_msg})

            print(f"  Turn {turn_idx + 1}: generating...")
            answer = generate_answer(model, tokenizer, conversation, config)
            answers.append(answer)
            conversation.append({"role": "assistant", "content": answer})
            print(f"  Turn {turn_idx + 1}: {len(answer)} chars")

        result = {
            "question_id": question["question_id"],
            "category": question["category"],
            "turns": question["turns"][:num_turns],
            "answers": answers,
        }

        # Judge if OpenAI key is available
        judgment = judge_answer(question, answers, judge_model)
        result["judgment"] = judgment
        if judgment["score"] > 0:
            print(f"  Score: {judgment['score']}/10")

        results.append(result)

    gen_time = time.time() - start

    # Compute aggregate scores
    category_scores = {}
    all_scores = []
    for r in results:
        score = r["judgment"].get("score", -1)
        if score > 0:
            all_scores.append(score)
            cat = r["category"]
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(score)

    category_avgs = {
        cat: round(sum(scores) / len(scores), 2)
        for cat, scores in category_scores.items()
    }
    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else -1

    elapsed = time.time() - start
    hours = elapsed / 3600

    # Write outputs
    with open(os.path.join(output_dir, "answers.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "task_id": os.environ.get("TASK_ID", ""),
        "model_id": model_id,
        "judge_model": judge_model,
        "num_questions": len(questions),
        "num_turns": num_turns,
        "overall_score": overall_avg,
        "category_scores": category_avgs,
        "num_judged": len(all_scores),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(hours, 3),
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*50}")
    print(f"MT-Bench Results: {model_id}")
    print(f"{'='*50}")
    if category_avgs:
        for cat, avg in sorted(category_avgs.items()):
            print(f"  {cat:15s}: {avg:.1f}/10")
        print(f"  {'overall':15s}: {overall_avg:.1f}/10")
    else:
        print("  No judge scores (OPENAI_API_KEY not set)")
        print("  Answers saved — run judgment separately")
    print(f"\nCompleted in {elapsed:.0f}s ({hours:.2f} GPU-hours)")


if __name__ == "__main__":
    main()
