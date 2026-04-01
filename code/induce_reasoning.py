import json
from argparse import ArgumentParser
import numpy as np
import random
import re
from datetime import datetime
from os.path import isfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd


INDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing inductive reasoning. You are given a set of observations. Your task is to infer the most probable general rule that explains all observations. Then, use the inferred rule to make predictions about new observations. Provide a detailed reasoning process leading to your conclusion."""

DEDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing deductive reasoning. You are given a general rule and specific observations. Your task is to apply the general rule to the observations to derive a logically certain conclusion. Provide a detailed reasoning process leading to your conclusion."""

ABDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing abductive reasoning. You are given a general rule and an observation. Your task is to generate the most probable and simplest hypothesis that, if true, would logically explain all the observations provided. Provide a detailed reasoning process leading to your conclusion."""

DIRECT_PROMPT = """Instruction: Please answer the question based on the given context directly without additional explanation or reasoning steps."""

UNCERTAINTY_PROMPT = """
Give the probability that you think your answer is correct as a value between 0.0 and 1.0. Take your uncertainty in the prompt, the task difficulty, your knowledge availability and other sources of uncertainty into account. Return the probability as <probability."""

THINKING_MODELS = ["Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "Qwen/Qwen3-8B"]


def init_transformers_model_and_tokenizer(model_name: str, seed: int, num_gpus: int, enforce_eager: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if enforce_eager:
        print("Warning: --enforce_eager is vLLM-specific and ignored with Transformers backend.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        # device_map='auto' supports both single- and multi-GPU setups.
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    if num_gpus > 1 and not torch.cuda.is_available():
        print("Warning: --num_gpus > 1 requested, but CUDA is unavailable. Falling back to CPU.")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def generate_chat_outputs(
    model,
    tokenizer,
    input_list,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
):
    if enable_thinking:
        print("Warning: enable_thinking is not directly supported by Transformers chat templates; proceeding without it.")

    if len(input_list) == 0:
        return []

    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in input_list
    ]

    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        generation_output = model.generate(**model_inputs, **generation_kwargs)

    sequences = generation_output.sequences
    scores = generation_output.scores
    prompt_lens = model_inputs["attention_mask"].sum(dim=1).tolist()

    results = []
    for i in range(len(input_list)):
        prompt_len = int(prompt_lens[i])
        generated_ids = sequences[i, prompt_len:]

        # Stop metrics at EOS when present to avoid counting padded tail tokens.
        generated_token_ids = []
        for token_id in generated_ids.tolist():
            generated_token_ids.append(token_id)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        cumulative_logprob = 0.0
        for t, token_id in enumerate(generated_token_ids):
            if t >= len(scores):
                break
            step_log_probs = torch.log_softmax(scores[t][i], dim=-1)
            cumulative_logprob += float(step_log_probs[token_id].item())

        if len(generated_token_ids) > 0:
            perplexity = float(np.exp(-cumulative_logprob / len(generated_token_ids)))
        else:
            perplexity = None

        if len(scores) > 0:
            first_step_probs = torch.softmax(scores[0][i], dim=-1)
            top_vals = torch.topk(first_step_probs, k=2).values
            margin = float((top_vals[0] - top_vals[1]).item())
        else:
            margin = None

        results.append({
            "text": text,
            "token_ids": generated_token_ids,
            "cumulative_logprob": cumulative_logprob,
            "perplexity": perplexity,
            "margin": margin,
        })

    return results


def correct_format(input_text: str) -> str:
    output_text = input_text.replace("\"True\"", "true")
    output_text = output_text.replace("\"False\"", "false")
    return output_text


def get_dataset(dataset_name: str):
    questions = []
    gold_answer_list = []
    reasoning_types = []
    id_list = []

    if dataset_name == "folio":
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
            question = f"\nQuestion: Is the conclusion True, False, or Uncertain based on the given premises. Premises: {premises} Conclusion: {conclusion}\nPlease enclose the answer in <answer></answer>."
            questions.append(question)
            gold_answer_list.append(gold_answer.lower())
            reasoning_types.append("deductive")
            if story_id not in id_dict:
                id_dict[story_id] = 0
            else:
                id_dict[story_id] += 1
            id_list.append(f"{story_id}_{id_dict[story_id]}")

    elif dataset_name == "folio_train":
        this_data = load_dataset(
                "yale-nlp/FOLIO",
                split="train"
                ).with_format("pandas")
        id_dict = {}
        for i in range(len(this_data)):
            premises = this_data["premises"][i]
            conclusion = this_data["conclusion"][i]
            gold_answer = this_data["label"][i]
            story_id = this_data["story_id"][i]
            question = f"\nQuestion: Is the conclusion True, False, or Uncertain based on the given premises. Premises: {premises} Conclusion: {conclusion}\nPlease enclose the answer in <answer></answer>."
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
            question = "\nQuestion: " + data_list[i]["context"].split(", and")[0] + "."
            question = question.replace("<answer><answer>", "<answer></answer>")
            questions.append(question)
            gold_answer_list.append(data_list[i]["answer"])
            reasoning_types.append("inductive")
            id_list.append(data_list[i]["id"])

    elif dataset_name == "number_array_train":
        with open("inductive/number_array/inductive_data_train.json", 'r') as f:
            data_list = json.load(f)
        for i in range(len(data_list)):
            question = "\nQuestion: " + data_list[i]["context"].split(", and")[0] + "."
            question = question.replace("<answer><answer>", "<answer></answer>")
            questions.append(question)
            gold_answer_list.append(data_list[i]["answer"])
            reasoning_types.append("inductive")
            id_list.append(data_list[i]["id"])

    elif "anli" in dataset_name:
        this_df = pd.read_csv(f"abductive/anli/{dataset_name}.csv")
        for i in range(len(this_df)):
            questions.append("\nQuestion: " + this_df["question"][i] + "\nPlease enclose the answer in <answer></answer>.")
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
    
    elif "folio" in dataset_name:
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
    
    elif dataset_name == "number_array" or dataset_name == "number_array_train":
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


def get_verbalize_confidence(response: str) -> float:
    prob_match = re.search(r'<probability>(.*?)</probability>|<(\d*\.?\d+)>', response)
    if prob_match:
        prob_str = prob_match.group(1) or prob_match.group(2)
        try:
            probability = float(prob_str)
            if 0.0 <= probability <= 1.0:
                return probability
            else:
                return -1.0
        except ValueError:
            return -1.0
    else:
        return -1.0


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--reasoning_type", type=str, default="none")
    parser.add_argument("--dataset", type=str, default="vitc")
    parser.add_argument("--vanilla", action="store_true", default=False)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="Reasoner")
    parser.add_argument("--think", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--verbalize_confidence", action="store_true", default=False)
    args = parser.parse_args()

    model_prefix = args.model_name.split("/")[-1]

    questions, gold_answer_list, reasoning_types, id_list = get_dataset(args.dataset)

    data_prefix = args.dataset.replace("/", "_").replace(".json", "")
    out_path = f"output/{model_prefix}_{args.seed}_{data_prefix}_{args.reasoning_type}.json"
    experiment_name = args.exp_name

    if isfile(out_path):
        with open(out_path, 'r') as f:
            output_file = json.load(f)
        correct_counts = [entry["correct"] for entry in output_file]
        tk_len_list = [entry["step2_num_tokens"] for entry in output_file]
    else:
        output_file = []
        tk_len_list = []
        correct_counts = []


    model, tokenizer = init_transformers_model_and_tokenizer(
        model_name=args.model_name,
        seed=args.seed,
        num_gpus=args.num_gpus,
        enforce_eager=args.enforce_eager,
    )

    input_list = []
    verbalized_confidence_input_list = []
    for i in range(len(output_file), len(questions)):

        if args.reasoning_type == "none":
            step2_prompt = questions[i]
        elif args.reasoning_type == "abductive":
            step2_prompt = ABDUCTIVE_PROMPT+questions[i]
        elif args.reasoning_type == "inductive":
            step2_prompt = INDUCTIVE_PROMPT+questions[i]
        elif args.reasoning_type == "deductive":
            step2_prompt = DEDUCTIVE_PROMPT+questions[i]
        elif args.reasoning_type == "direct":
            step2_prompt = DIRECT_PROMPT+questions[i]

        input_list.append([
                {"role": "user", "content": step2_prompt}
        ])
        verbalized_confidence_input_list.append([
                {"role": "user", "content": step2_prompt + UNCERTAINTY_PROMPT}
        ])

    output = generate_chat_outputs(
        model=model,
        tokenizer=tokenizer,
        input_list=input_list,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.think and args.model_name in THINKING_MODELS,
    )
    
    if args.verbalize_confidence:
        confident_output = generate_chat_outputs(
            model=model,
            tokenizer=tokenizer,
            input_list=verbalized_confidence_input_list,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.think and args.model_name in THINKING_MODELS,
        )

    o_len = len(output_file)
    for i in range(o_len, len(questions)):
        output_id = i - o_len
        step2_response = output[output_id]["text"]
        tk_ids = output[output_id]["token_ids"]
        cumulative_logprob = output[output_id]["cumulative_logprob"]
        perplexity = output[output_id]["perplexity"]
        margin = output[output_id]["margin"]

        # logprobs = output[output_id].outputs[0].logprobs

        if args.verbalize_confidence:
            confidence_response = confident_output[output_id]["text"]
            verbalized_confidence = get_verbalize_confidence(confidence_response)
        else:
            confidence_response = None
            verbalized_confidence = None

        if type(gold_answer_list[i]) == str:
            gold_answer = gold_answer_list[i]
        else:
            gold_answer = str(gold_answer_list[i])


        if type(step2_response) == str:
            correct = is_correct(step2_response.lower(), gold_answer, dataset_name=args.dataset)
        else:
            correct = 0

        correct_counts.append(correct)
        output_file.append({
              "id": id_list[i],
              "reasoning_type": reasoning_types[i],
              "step2_prompt": input_list[output_id][0]["content"],
              "step2_response": step2_response,
              "step2_num_tokens": len(tk_ids),
              "gold_answer": gold_answer,
              "correct": correct,
              "perplexity": perplexity,
              "margin": margin, # first step margin
              "verbalized_confidence": verbalized_confidence,
              "confidence_response": confidence_response
            })
        tk_len_list.append(len(tk_ids))
    
    accuracy = sum(correct_counts) / len(correct_counts) * 100
    avg_tokens = sum(tk_len_list) / len(tk_len_list)

    with open(out_path, 'w') as f:
        json.dump(output_file, f, indent=4)

    if isfile(f"aggregated_results/{experiment_name}_results.json"):
        with open(f"aggregated_results/{experiment_name}_results.json", 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    if args.model_name not in existing_results:
        existing_results[args.model_name] = {}
    
    if args.dataset not in existing_results[args.model_name]:
        existing_results[args.model_name][args.dataset] = {}
    
    if args.reasoning_type not in existing_results[args.model_name][args.dataset]:
        existing_results[args.model_name][args.dataset][args.reasoning_type] = {
            "Accuracy": [],
            "Avg_Tokens": []
        }
    

    existing_results[args.model_name][args.dataset][args.reasoning_type]["Accuracy"].append(accuracy)
    existing_results[args.model_name][args.dataset][args.reasoning_type]["Avg_Tokens"].append(avg_tokens)
    

    with open(f"aggregated_results/{experiment_name}_results.json", 'w') as f:
        json.dump(existing_results, f, indent=4)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average number of tokens: {avg_tokens:.2f}")
    end_time = datetime.now()
    print(f"Finishing script at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - start_time
    print(f"Time spent: {time_diff}")
