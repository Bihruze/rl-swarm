from typing import List 
import os
import random
import re

import numpy as np

import hivemind_exp.gsm8k.stage1_rewards as stage1_rewards
from hivemind_exp.hivemind_utils import HivemindNode


def extract_xml_identity(text: str) -> str:
    id = text.split("<identify>")[-1]
    id = id.split("</identify>")[0]
    return id.strip()


def extract_xml_ids(text: str) -> List[str]:
    if text is None:
        return []
    ids_raw = text.split("<student>")[1:]
    return [id_10.split("</student>")[0] for id_10 in ids_raw]

def extract_original_question(text: str) -> str:
    if text is None:
        return ""
    q = text.split("### The following answers to this question were given is: ")[-1]
    return q

def extract_answers(text: str) -> List[str]:
    if text is None:
        return []
    answers_raw = text.split("<student>")[1:].strip()
    answers = [a.split("</student>")[0].strip() for a in answers_raw if a != "\n"]
    return answers


def count_xml(text) -> float:
    count = 0.0
    if text.count("<compare>\n") == 1:
        count += 0.125
    if text.count("\n</compare>\n") == 1:
        count += 0.125
    if text.count("<explain>\n") == 1:
        count += 0.125
    if text.count("\n</explain>\n") == 1:
        count += 0.125
    if text.count("\n<identify>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</identify>\n")[-1]) * 0.001
    if text.count("\n</identify>") == 1:
        count += 0.125
        count -= (len(text.split("\n</identify>")[-1]) - 1) * 0.001
    return count


# Reward functions
def proper_id_reward_func(
    prompts, completions, answer, weighting=2.0, logging=True, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    p = prompts[0][-1]["content"]
    agent_ids = extract_xml_ids(p)
    extracted_responses = [extract_xml_identity(r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "id_extact_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nValid IDs:\n{agent_ids}\n\nExtracted:\n{extracted_responses[0]}\n\nGot reward? {extracted_responses[0] in agent_ids}"
            f.write(out_line)
    return [1.0 * weighting if r in agent_ids else 0.0 for r in extracted_responses]


def correctness_reward_func(
    prompts, completions, answer, weighting=2.0, logging=True, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    p = prompts[0][-1]["content"]
    agent_answers = extract_answers(p)
    extracted_responses = [extract_xml_identity(r) for r in responses]
    chosen_rewards = []
    for r in extracted_responses:
        cur_reward = 0
        if r in agent_answers:
            if stage1_rewards.extract_xml_answer(agent_answers[r]) == answer[0]:
                cur_reward += 1.0
            if stage1_rewards.extract_xml_answer(agent_answers[r]).isdigit():
                cur_reward += 0.5
            pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 0.5
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
            if re.match(pattern, agent_answers[r]):
                cur_reward += 0.5
            cur_reward += stage1_rewards.count_xml(agent_answers[r])
        elif r in [
            "None",
            "No one",
            "All answers are wrong",
            "All answers were wrong",
            "All are wrong",
            "All were wrong",
            "None are correct",
            "None were correct",
            "No one is correct",
        ]:
            agent_as = [
                stage1_rewards.extract_xml_answer(agent_answers[id])
                for id in agent_answers
            ]
            check_submissions = [
                True if r == a else False for r, a in zip(agent_as, answer)
            ]
            if all(check_submissions):
                cur_reward += 10
        chosen_rewards += [cur_reward]
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        if extracted_responses[0] in agent_answers:
            os.makedirs(
                f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                exist_ok=True,
            )
            log_file = os.path.join(
                "model_output_samples",
                f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
                "correctness_samps.txt",
            )
            with open(log_file, "a") as f:
                f.write("-" * 20)
                out_line = f"\nPrompt:\n{p}\n\nResponse:\n{responses[0]}\n\nChosen answer ID:\n{extracted_responses[0]}\n\nExtracted:\n{agent_answers[extracted_responses[0]]}\n\nReward for choice: {chosen_rewards[0]}"
                f.write(out_line)
    return [r * weighting for r in chosen_rewards]


def strict_format_reward_func(
    completions, weighting=0.5, logging=True, **kwargs
) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<compare>\n.*?\n</compare>\n<explain>\n.*?\n</explain>\n<identify>\n.*?\n</identify>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s2_strict_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    return [1.0 * weighting if match else 0.0 for match in matches]


def soft_format_reward_func(
    completions, weighting=0.5, logging=True, **kwargs
) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = (
        r"<compare>.*?</compare>\s*<explain>.*?</explain>\s*<identify>.*?</identify>"
    )
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "s2_soft_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = f"\nResponse:\n{responses[0]}\n\nMatches? {matches[0]}"
            f.write(out_line)
    return [1.0 * weighting if match else 0.0 for match in matches]


def xmlcount_reward_func(
    completions, weighting=1.0, logging=True, **kwargs
) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    if (random.random() < 0.01) and logging:  # 1% chance to write samples into a file
        os.makedirs(
            f"model_output_samples/multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            exist_ok=True,
        )
        log_file = os.path.join(
            "model_output_samples",
            f"multi_stage_gsm8k_samples_from_{os.getenv('HOSTNAME')}",
            "strict_format_samps.txt",
        )
        with open(log_file, "a") as f:
            f.write("-" * 20)
            out_line = (
                f"\nResponse:\n{contents[0]}\n\nCount reward: {count_xml(contents[0])}"
            )
            f.write(out_line)
    return [count_xml(c) * weighting for c in contents]

def top_k_cumulative_reward(
    prompts,
    completions,
    answer,
    logging=False,
    **kwargs,
) -> list[float]:
    """
    Dummy reward function that accumulates all rewards into one for prompt generation's top_k selector
    """
    proper_id_reward = proper_id_reward_func(
        prompts, completions, answer, logging=logging
    )
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    strict_format_reward = strict_format_reward_func(completions, logging=logging)
    soft_format_reward = soft_format_reward_func(completions, logging=logging)
    xmlcount_reward = xmlcount_reward_func(completions, logging=logging)
    total_reward = [
        sum(tup)
        for tup in zip(
            proper_id_reward,
            correctness_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]
    return total_reward


def hivemind_cumulative_reward(
    node: HivemindNode,
    prompts,
    completions,
    answer,
    logging=False,
    output_signal_selector="max",
    **kwargs,
) -> list[float]:
    """
    Dummy reward function that accumulates all rewards into one + saves JSON to node.outputs
    """
    proper_id_reward = proper_id_reward_func(
        prompts, completions, answer, logging=logging
    )
    correctness_reward = correctness_reward_func(
        prompts, completions, answer, logging=logging
    )
    strict_format_reward = strict_format_reward_func(completions, logging=logging)
    soft_format_reward = soft_format_reward_func(completions, logging=logging)
    xmlcount_reward = xmlcount_reward_func(completions, logging=logging)
    total_reward = [
        sum(tup)
        for tup in zip(
            proper_id_reward,
            correctness_reward,
            strict_format_reward,
            soft_format_reward,
            xmlcount_reward,
        )
    ]

    question = extract_original_question(prompts[0][-1]["content"])
    if output_signal_selector == "max":
        # Generate output line
        maximal_reward_idx, responses = (
            np.argmax(total_reward),
            [completion[0]["content"] for completion in completions],
        )
        output_data = {
            "question": question,
            "answer": answer[0],
            "stage2_prompt": prompts[0][-1]["content"],
            "agent_opinion": {node.key: responses[maximal_reward_idx]},
        }

    if output_signal_selector != None:
        node.outputs = output_data
        node.rewards = total_reward

    return [0.0 for _ in total_reward]
