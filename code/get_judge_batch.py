"""
generating a batch API file as the following format:
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

"""
import json
from argparse import ArgumentParser
from os.path import isfile
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join


SYSTEM_PROMPT = "You are a helpful assistant."
PROCESS_QUESTION="What type of reasoning is the following process?"
PROCESS_QUESTION_2="""Instruction: Please classify the type of reasoning demonstrated in the following process.
The following are definitions of the three foundamental reasoning types:
- Deductive reasoning: Deductive reasoning is the process of reasoning from one or more general statements (premises) to reach a logically certain conclusion. It involves applying established principles or rules to specific cases to derive conclusions that must be true if the premises are true.
- Inductive reasoning: Inductive reasoning is the process of reasoning in which a set of general principles or rules is drawn from the given observations. It involves identifying patterns in the observations and using them to formulate broader generalizations or theories that must be true on all the observations. The final answer is acquired after applying the rule.
- Abductive reasoning: Abductive reasoning is the process of reasoning that involves forming a hypothesis or explanation based on the available evidence or observations. Based on established principles or rules, abductive reasoning generates the most plausible explanations for the observations even though there is incomplete information.

The key difference between inductive reasoning and abductive reasoning is that the inductive reasoning gets the answer by applying the rule, while abductive reasoning compares the possible answers and gets the most possible one which is often not certain.
If there is no reasoning process involved, please respond with **direct**. Otherwise, please respond with one of the reasoning types: **deductive**, **inductive**, or **abductive**.
"""
DIRECT_PROMPT = """Instruction: Please answer the question based on the given context directly without additional explanation or reasoning steps."""


def get_trajectory_data(data_path):
    questions = []
    id_list = []
    for file in listdir(data_path):
        if file.endswith(".json"):
            print(file)

            id_head = file.replace("single_type_", "").replace(".json", "")
            if isfile(join(data_path, file)):
                with open(join(data_path, file), 'r') as f:
                    this_data = json.load(f)
                for i in range(len(this_data)):
                    questions.append(f"{PROCESS_QUESTION}\nReasoning process:\n{this_data[i]['response']}")
                    # questions.append(f"{PROCESS_QUESTION_2}\nReasoning process:\n{this_data[i]['response']}")
                    id_list.append(f"{id_head}_{i}")
    return questions, id_list    
    

def get_gpt_trajectory_data(data_path):
    questions = []
    id_list = []
    for file in listdir(data_path):
        if file.startswith("single_type_") and file.endswith("_response.jsonl"):
            # if not "_recv_" in file:
            #      continue
            if "_none" in file:
                continue
            if "_direct" in file:
                continue

            print(file)
            count = 0
            id_head = file.replace("single_type_", "").replace("_response.jsonl", "")
            this_data_path = join(data_path, file)
            if isfile(this_data_path):
                with open(this_data_path, 'r') as f:
                    for line in f.readlines():
                        output = json.loads(line)
                        decoded_output = output["response"]["body"]["choices"][0]["message"]["content"].strip()

                        questions.append(f"{PROCESS_QUESTION}\nReasoning process:\n{decoded_output}")
                        # questions.append(f"{PROCESS_QUESTION_2}\nReasoning process:\n{this_data[i]['response']}")
                        id_list.append(f"{id_head}_{count}")
                        count += 1
    return questions, id_list   


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5.1")
    parser.add_argument("--data_path", type=str, default="output_files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    if "gpt" in args.data_path:
        questions, id_list = get_gpt_trajectory_data(args.data_path)
    else:
        questions, id_list = get_trajectory_data(args.data_path)
        

    m_name = args.model_name.replace(".", "")
    d_name = args.data_path.split("/")[-1].replace(".csv", "")
    save_path = f"batch_api/{m_name}_{d_name}_{args.seed}.jsonl"

    out_str = ""
        
    print(f"Generating batch API file for {len(questions)} questions.")
    for i in range(len(questions)):
        question = questions[i]
        if "gpt" in args.model_name:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]

            this_call = {
                "custom_id": f"{m_name}_{d_name}_{args.seed}_{id_list[i]}",
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
                "key": f"{m_name}_{d_name}_{args.seed}_{id_list[i]}",
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
    with open(save_path, 'w') as f:
        f.write(out_str)

    print(f"Batch API file saved to {save_path}")
    

        