import json
from argparse import ArgumentParser
import numpy as np
import random
import re
from datetime import datetime
from os.path import isfile
from datasets import load_dataset
import pandas as pd


SYSTEM_PROMPT = "You are a helpful assistant."

INDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing inductive reasoning. You are given a set of observations. Your task is to infer the most probable general rule that explains all observations. Then, use the inferred rule to make predictions about new observations. Provide a detailed reasoning process leading to your conclusion."""

DEDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing deductive reasoning. You are given a general rule and specific observations. Your task is to apply the general rule to the observations to derive a logically certain conclusion. Provide a detailed reasoning process leading to your conclusion."""

ABDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing abductive reasoning. You are given a general rule and an observation. Your task is to generate the most probable and simplest hypothesis that, if true, would logically explain all the observations provided. Provide a detailed reasoning process leading to your conclusion."""

DIRECT_PROMPT = """Instruction: Please answer the question based on the given context directly without additional explanation or reasoning steps."""


def correct_format(input_text: str) -> str:
    output_text = input_text.replace("\"True\"", "true")
    output_text = output_text.replace("\"False\"", "false")
    return output_text


def get_dataset(dataset_name: str):
    questions = []
    gold_answer_list = []
    reasoning_types = []
    id_list = []
    if dataset_name == "vitc" or dataset_name == "climate_fever" or dataset_name == "phemeplus":
        with open(f"mix/RECV/{dataset_name}_500_with_types.json", 'r') as f:
            data_list = json.load(f)
        
        
        for i in range(len(data_list)):
            context = data_list[i]["context"]
            question = data_list[i]["question"]
            gold_answer = "True" if data_list[i]["answer"]=="A" else "False"
            gold_answer_list.append(gold_answer.lower())

            question = f"\nQuestion: {context}\n{question}\nPlease enclose the answer in <answer><answer>."
            questions.append(question)
            reasoning_types.append(data_list[i]["reasoning_type"])
            id_list.append(data_list[i]["id"])

    elif dataset_name == "folio":
        this_data = load_dataset(
                "yale-nlp/FOLIO",
                split="validation"
                ).with_format("pandas")
        id_dict = {}
        for i in range(len(this_data)):
            premises = this_data["premises"][i]
            conclusion = this_data["conclusion"][i]
            gold_answer = this_data["label"][i]
            story_id = this_data["story_id"][i]
            question = f"\nQuestion: Is the conclusion True, False, or Uncertain based on the given premises. Premises: {premises} Conclusion: {conclusion}\nPlease enclose the answer in <answer><answer>."
            questions.append(question)
            gold_answer_list.append(gold_answer.lower())
            reasoning_types.append("deductive")
            if story_id not in id_dict:
                id_dict[story_id] = 0
            else:
                id_dict[story_id] += 1
            id_list.append(f"{story_id}_{id_dict[story_id]}")
    
    elif dataset_name == "number_array":
        with open("inductive/number_array/inductive_data_test.json", 'r') as f:
            data_list = json.load(f)
        for i in range(len(data_list)):
            questions.append("\nQuestion: " + data_list[i]["context"].split(", and")[0] + ".")
            gold_answer_list.append(data_list[i]["answer"])
            reasoning_types.append("inductive")
            id_list.append(data_list[i]["id"])

    elif "anli" in dataset_name:
        this_df = pd.read_csv(f"abductive/anli/{dataset_name}.csv")
        for i in range(len(this_df)):
            questions.append("\nQuestion: " + this_df["question"][i] + "\nPlease enclose the answer in <answer><answer>.")
            gold_answer_list.append(this_df["answer"][i].lower())
            reasoning_types.append("abductive")
            id_list.append(this_df["id"][i])

    elif "recv" in dataset_name:
        data_df = pd.read_csv(f"mix/RECV/{dataset_name}.csv")

        for i in range(len(data_df)):
            question = data_df["question"][i] + "\nPlease enclose the answer in <answer></answer>."
            gold_answer = data_df["answer"][i]
            gold_answer_list.append(str(gold_answer).lower())
            questions.append(question)
            reasoning_types.append(data_df["reasoning_type"][i])
            id_list.append(data_df["id"][i])
        
    return questions, gold_answer_list, reasoning_types, id_list


def is_correct(response: str, gold_answer: str, dataset_name: str = "recv") -> int:
    if dataset_name == "sat":
        answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', response, re.DOTALL)

        if answer_match:
            predicted_answer = answer_match.group(1) or answer_match.group(2)
            try:
                predicted_answer_dict = json.loads(predicted_answer)
            except json.JSONDecodeError:
                return 0
        else:
            return 0

        gold_answer = correct_format(gold_answer)
        gold_answer_dict = json.loads(gold_answer)

        all_correct = True
        for key in gold_answer_dict:
            if key in predicted_answer_dict:
                if predicted_answer_dict[key] != gold_answer_dict[key]:
                    all_correct = False
                    break
            else:
                all_correct = False
                break

        if all_correct:
            return 1
        else:
            return 0
    
    elif dataset_name == "folio":
        answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', response, re.DOTALL)

        if answer_match:
            predicted_answer = answer_match.group(1) or answer_match.group(2)
            if predicted_answer is not None:
                predicted_answer = predicted_answer.strip()
            else:
                return 0
        else:
            return 0

        if re.search(r'\btrue\b', predicted_answer):
            predicted_answer_bool = "true"
        elif re.search(r'\bfalse\b', predicted_answer):
            predicted_answer_bool = "false"
        elif re.search(r'\buncertain\b', predicted_answer):
            predicted_answer_bool = "uncertain"
        else:
            return 0

        if predicted_answer_bool == gold_answer:
            return 1
        else:
            return 0
    
    elif dataset_name == "number_array":
        answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', response, re.DOTALL)

        if answer_match:
            predicted_answer = answer_match.group(1) or answer_match.group(2)
            if predicted_answer is not None:
                predicted_answer = predicted_answer.strip()
        else:
            return 0

        if predicted_answer == gold_answer:
            return 1
        else:
            return 0
    
    elif "anli" in dataset_name:
        answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', response, re.DOTALL)

        if answer_match:
            predicted_answer = answer_match.group(1) or answer_match.group(2)
            if predicted_answer is not None:
                predicted_answer = predicted_answer.strip()
            else:
                return 0
        else:
            return 0
        
        gold_answer = gold_answer[:-1]
        if predicted_answer.endswith('.'):
            predicted_answer = predicted_answer[:-1]

        if predicted_answer == gold_answer:
            return 1
        else:
            return 0

    elif "recv" in dataset_name:
        answer_match = re.search(r'<answer>(.*?)</answer>|<answer>(.*?)<answer>', response, re.DOTALL)

        if answer_match:
            predicted_answer = answer_match.group(1) or answer_match.group(2)
            if predicted_answer is not None:
                predicted_answer = predicted_answer.strip()
            else:
                return 0
        else:
            return 0

        if re.search(r'\btrue\b', predicted_answer):
            predicted_answer_bool = "true"
        elif re.search(r'\bfalse\b', predicted_answer):
            predicted_answer_bool = "false"
        else:
            return 0

        if predicted_answer_bool == gold_answer:
            return 1
        else:
            return 0


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5.1")
    parser.add_argument("--reasoning_type", type=str, default="none")
    parser.add_argument("--dataset", type=str, default="vitc")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_prefix = args.model_name.split("/")[-1]

      
    questions, gold_answer_list, reasoning_types, id_list = get_dataset(args.dataset)

    data_prefix = args.dataset.replace("/", "_").replace(".json", "")
    out_path = f"batch_api/{model_prefix}_{args.seed}_{data_prefix}_{args.reasoning_type}.jsonl"

    out_str = ""


    for i in range(len(questions)):
        if args.reasoning_type == "none":
            question = questions[i] + "\nPlease also return step-by-step reasoning process in your response."
        elif args.reasoning_type == "abductive":
            question = ABDUCTIVE_PROMPT+questions[i]
        elif args.reasoning_type == "inductive":
            question = INDUCTIVE_PROMPT+questions[i]
        elif args.reasoning_type == "deductive":
            question = DEDUCTIVE_PROMPT+questions[i]
        elif args.reasoning_type == "direct":
            question = DIRECT_PROMPT+questions[i]

        if "gpt" in args.model_name:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]

            this_call = {
                "custom_id": f"{model_prefix}_{data_prefix}_{args.seed}_{args.reasoning_type}_{id_list[i]}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model_name,
                    "messages": messages,
                    "temperature": args.temperature,
                    "seed": args.seed
                }
            }
        elif "gemini" in args.model_name:
            this_call = {
                "key": f"{model_prefix}_{data_prefix}_{args.seed}_{args.reasoning_type}_{id_list[i]}",
                "request": {
                    "contents": [
                        {
                            "parts": [
                                {"text": question}
                            ]
                        }
                    ],
                    "generation_config": {
                        "temperature": args.temperature,
                        "seed": args.seed
                    },
                }
            }

        out_str += json.dumps(this_call) + "\n"
    
    out_str = out_str[:-1]  # remove the last newline character
    with open(out_path, 'w') as f:
        f.write(out_str)

    print(f"Batch API file saved to {out_path}")