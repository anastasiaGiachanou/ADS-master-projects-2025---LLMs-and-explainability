#!/usr/bin/env python3
"""
GSM8K Faithfulness Evaluation using Gemma 3 27B as Judge
Returns faithfulness score (0-1)
"""

import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel

# Configuration
FINETUNED_MODEL_PATH = "finetuned-model-path"
JUDGE_MODEL_NAME = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
NUM_SAMPLES = 200
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
QUESTION_BATCH_SIZE = 16
JUDGE_BATCH_SIZE = 16


def setup_judge_model():
    judge_model, judge_tokenizer = FastLanguageModel.from_pretrained(
        model_name=JUDGE_MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(judge_model)
    return judge_model, judge_tokenizer


def setup_target_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_MODEL_PATH,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


SYSTEM_PROMPT = """You are a precise arithmetic assistant.
For each question, output exactly:

<reasoning>
…your step-by-step operations…
</reasoning>
<answer>
…your final numeric answer…
</answer>"""


def format_prompt(question: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\nLet's think step by step.\n<reasoning>\n"
    )


def extract_reasoning(text: str) -> str:
    try:
        # Look for reasoning between tags
        blocks = re.findall(
            r"<reasoning>\s*([\s\S]*?)\s*</reasoning>", text, re.IGNORECASE | re.DOTALL
        )
        return blocks[-1].strip() if blocks else ""
    except:
        return ""


def extract_numeric_answer(text: str) -> str:
    try:
        # XML format
        xml_blocks = re.findall(
            r"<answer>\s*([\s\S]*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
        )
        if xml_blocks:
            content = xml_blocks[-1].strip()
            # Remove currency symbols and commas, then find number
            content = (
                content.replace("$", "")
                .replace(",", "")
                .replace("₹", "")
                .replace("€", "")
            )
            match = re.search(r"-?\d+(?:\.\d+)?", content)
            if match:
                return match.group(0)

        # Hash format fallback
        if "####" in text:
            hash_content = text.split("####")[1].strip()
            match = re.search(r"-?\d+(?:\.\d+)?", hash_content.replace(",", ""))
            if match:
                return match.group(0)

        return None
    except:
        return None


def extract_hash_answer(text: str) -> str:
    # Extract answer after '#### X' marker in ground truth
    if not isinstance(text, str) or "####" not in text:
        return None

    parts = text.split("####")
    if len(parts) < 2:
        return None

    answer_text = parts[1].strip()
    answer_text = answer_text.replace("$", "").replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", answer_text)
    return match.group(0) if match else None


def generate_reasoning_batch(questions: list, model, tokenizer):
    # Generate reasoning for a batch of questions
    prompts = [format_prompt(q) for q in questions]

    batch_inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs.get("attention_mask", None),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(questions)):
        decoded = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        response = decoded.split("<|assistant|>")[-1].strip()

        reasoning = extract_reasoning(response)
        answer = extract_numeric_answer(response)

        results.append((reasoning, answer))

    return results


def evaluate_faithfulness_batch(batch_data: list, judge_model, judge_tokenizer):
    judge_prompts = []

    for problem, reasoning, answer, correct_answer in batch_data:
        if not reasoning or not answer:
            judge_prompts.append(None)
            continue

        # Custom prompt for judge model
        judge_prompt = f"""<bos><start_of_turn>user
Evaluate this mathematical solution on a scale of 1-5:

Problem: {problem}

Student Reasoning: {reasoning}

Student Answer: {answer}

Correct Answer: {correct_answer}

Scale:
1 = Poor reasoning AND wrong answer
2 = Some logical steps but significant issues OR good reasoning with wrong answer
3 = Generally sound reasoning with minor issues, answer may be wrong due to arithmetic errors
4 = Strong reasoning with very minor flaws, correct or nearly correct answer
5 = Excellent reasoning with correct answer

Consider both the logical flow of reasoning AND the final answer accuracy.

Respond with only a number from 1 to 5.<end_of_turn>
<start_of_turn>model
"""
        judge_prompts.append(judge_prompt)

    # Filter for None prompt and indices
    valid_prompts = []
    valid_indices = []
    for i, prompt in enumerate(judge_prompts):
        if prompt is not None:
            valid_prompts.append(prompt)
            valid_indices.append(i)

    results = [1] * len(batch_data)

    if not valid_prompts:
        return results

    inputs = judge_tokenizer(
        valid_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = judge_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=10,
            temperature=1.0,
            top_k=64,
            top_p=0.95,
            do_sample=True,
            pad_token_id=judge_tokenizer.eos_token_id,
            eos_token_id=judge_tokenizer.eos_token_id,
        )

    for i, output_ids in enumerate(outputs):
        response = judge_tokenizer.decode(output_ids, skip_special_tokens=True)

        # Find the very last digit in the response
        all_digits = re.findall(r"[1-5]", response)
        if all_digits:
            score = int(all_digits[-1])
            results[valid_indices[i]] = score
        else:
            results[valid_indices[i]] = 3

    return results


def check_accuracy(pred_answer: str, ground_truth: str) -> bool:
    # Check if predicted answer matches ground truth
    if pred_answer is None or ground_truth is None:
        return False

    try:
        pred_num = float(pred_answer)
        gt_num = float(ground_truth)
        return abs(pred_num - gt_num) < 1e-6
    except ValueError:
        return pred_answer.strip() == ground_truth.strip()


def evaluate_dataset_batched(
    df, target_model, target_tokenizer, judge_model, judge_tokenizer
):
    faithfulness_scores = []
    accuracy_scores = []

    # Process multiple questions
    for batch_start in tqdm(range(0, len(df), QUESTION_BATCH_SIZE), desc="Evaluating"):
        batch_end = min(batch_start + QUESTION_BATCH_SIZE, len(df))
        batch_questions = []
        batch_ground_truths = []

        for i in range(batch_start, batch_end):
            sample = df.iloc[i]
            batch_questions.append(sample["question"])
            batch_ground_truths.append(extract_hash_answer(sample["answer"]))

        # Generate reasoning for all questions in batch
        batch_results = generate_reasoning_batch(
            batch_questions, target_model, target_tokenizer
        )

        judge_batch_data = []
        batch_accuracy = []

        for j, ((reasoning, predicted_answer), ground_truth) in enumerate(
            zip(batch_results, batch_ground_truths)
        ):
            is_correct = check_accuracy(predicted_answer, ground_truth)
            batch_accuracy.append(1.0 if is_correct else 0.0)

            judge_batch_data.append(
                (batch_questions[j], reasoning, predicted_answer, ground_truth)
            )

        batch_faithfulness = []
        for judge_batch_start in range(0, len(judge_batch_data), JUDGE_BATCH_SIZE):
            judge_batch_end = min(
                judge_batch_start + JUDGE_BATCH_SIZE, len(judge_batch_data)
            )
            judge_sub_batch = judge_batch_data[judge_batch_start:judge_batch_end]

            sub_batch_scores = evaluate_faithfulness_batch(
                judge_sub_batch, judge_model, judge_tokenizer
            )
            batch_faithfulness.extend(sub_batch_scores)

        faithfulness_scores.extend(batch_faithfulness)
        accuracy_scores.extend(batch_accuracy)

    return faithfulness_scores, accuracy_scores


def main():
    # Models setup
    judge_model, judge_tokenizer = setup_judge_model()
    target_model, target_tokenizer = setup_target_model()

    # Load dataset - GSM8K test set downloaded as a parquet file
    gsm8k = pd.read_parquet("test-00000-of-00001.parquet")
    test_data = gsm8k.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

    # Evaluate
    faithfulness_scores, accuracy_scores = evaluate_dataset_batched(
        test_data, target_model, target_tokenizer, judge_model, judge_tokenizer
    )

    # Calculate final score (convert 1-5 scale to 0-1 scale)
    avg_faithfulness = np.mean(faithfulness_scores)
    avg_accuracy = np.mean(accuracy_scores)

    faithfulness_score = (avg_faithfulness - 1) / 4

    # Weighted combination: 70% reasoning + 30% accuracy
    combined_score = faithfulness_score * 0.7 + avg_accuracy * 0.3

    # Print both scores
    print(f"Reasoning score: {faithfulness_score:.3f}")
    print(f"Weighted score: {combined_score:.3f}")

    return combined_score


if __name__ == "__main__":
    score = main()
    print(f"Final Score: {score:.3f}")
