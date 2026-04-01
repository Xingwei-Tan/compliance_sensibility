import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
import json
import re
from argparse import ArgumentParser
from os import listdir
from os.path import join, isfile 


LABELS= ["direct", "deductive", "inductive", "abductive"]
LABELS_MAP = {"no": 0, "deductive": 1, "inductive": 2, "abductive": 3, "direct": 0}
IGNORE = []

def get_base_id(full_id: str, prefix_len: int=4) -> str:
    base_id = "_".join(full_id.split("_")[prefix_len:])
    return base_id

def get_prediction(response: str) -> str:
    for matched in re.finditer(r'\*\*(.*?)\*\*', response):
        matched_text = matched.group(1).lower()
        for label in LABELS:
            if label in matched_text:
                return label
    response = response.strip().lower()
    if response in LABELS:
        return response
    else:
        return "direct"
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge_file", type=str, required=True, help="Path to the JSONL file with judge results.")
    parser.add_argument("--result_dir", type=str, default="confidence_output")
    parser.add_argument("--provider", type=str, default="openai", choices=["gemini", "openai"])
    args = parser.parse_args()


    file_ids = set()
    rows = []
    missing_content = []
    key_dict = set()
    prefix_len = len(args.judge_file.split("/")[-1].split("_")) - 1
    with open(args.judge_file, "r") as f:
        for line in f:
            data_dict = json.loads(line)
            if args.provider == "gemini":
                if len(data_dict["response"]["candidates"][0]["content"]) == 0:
                    continue
                else:
                    decoded_output = data_dict["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                key = get_base_id(data_dict.get("key"), prefix_len=prefix_len)
            elif args.provider == "openai":
                response = data_dict.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])
                message = choices[0].get("message", {}) 
                decoded_output = message.get("content", "").strip()
                key = get_base_id(data_dict.get("custom_id"), prefix_len=prefix_len)

            if key in key_dict:
                continue
            else:
                key_dict.add(key)
            row = {
                "base_id": key,
                "openai_reasoning_type": get_prediction(decoded_output),
                "openai_judge_output": decoded_output
            }
            rows.append(row)

            file_id = "_".join(key.split("_")[:-1])
            file_ids.add(file_id)
    

    table_df = pd.DataFrame(rows)
    print("openai judge response length:", len(table_df))

    print("File IDs found:", file_ids)
    print(len(file_ids))
    all_dfs = []
    for file_id in file_ids:
        if file_id in IGNORE:
            print(f"Ignoring file ID: {file_id}")
            continue
        file_path = join(args.result_dir, f"{file_id}.json")
        dataset = "_".join(file_id.split("_")[2:-1])
        model_name = file_id.split("_")[0]
        random_seed = file_id.split("_")[1]
        instructed_reasoning_type = file_id.split("_")[-1]

        other_df = pd.read_json(file_path)

        base_id_list = []
        for i in range(len(other_df)):
            base_id_list.append(f"{file_id}_{i}")

        other_df["base_id"] = base_id_list
        other_df["dataset"] = dataset
        other_df["model_name"] = model_name
        other_df["random_seed"] = random_seed
        other_df["instructed_reasoning_type"] = instructed_reasoning_type
        all_dfs.append(other_df)

    other_df = pd.concat(all_dfs, ignore_index=True)
    table_df = table_df.merge(other_df, on="base_id", how="inner")
    table_df["reasoning_match_question"] = table_df["openai_reasoning_type"] == table_df["reasoning_type"]
    table_df["reasoning_match_instruction"] = table_df["openai_reasoning_type"] == table_df["instructed_reasoning_type"]
    table_df["correctness"] = table_df["correct"] == 1
    table_df = table_df.drop(columns=["base_id", "correct", "correctness", "reasoning_type", "openai_judge_output", "step2_prompt", "step2_response"])
    table_df = table_df.rename(columns={"correctness": "correct"})

    instruction_outcome = []
    for i in range(len(table_df)):
        instruction_outcome.append(f"{table_df["reasoning_match_question"][i]}_{table_df["reasoning_match_instruction"][i]}")
    table_df["instruction_outcome"] = instruction_outcome
        

    print("Final table length:", len(table_df))
    table_df.to_csv("final_tables/results.csv", index=False)