"""
This is a short test on getting the steering data
The formal data should be built from training set
"""

import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
import json
import re
from argparse import ArgumentParser
from os import listdir
from os.path import join, isfile 



    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--table_path", type=str, required=True, help="Path to the csv file with all the results.")
    parser.add_argument("--model_name", type=str, default="Llama-3.3-70B-Instruct")
    parser.add_argument("--dataset", type=str, default="anli_test")
    args = parser.parse_args()

    dataset_name = f"{args.model_name}_{args.dataset}_compliance"
    dataset = {dataset_name:[]}

    positive_negative_dict = {} 


    df = pd.read_csv(args.table_path)

    df = df[df["model_name"] == args.model_name]
    if args.dataset != "all":
        df = df[df["dataset"] == args.dataset]
    df = df.reset_index(drop=True)


    for i in range(len(df)):
        this_id = df["id"][i]
        this_output = df["model_output"][i]
        compliance = df["reasoning_match_instruction"][i]
        this_actual_reasoning = df["gemini_reasoning_type"][i]
        this_instructed_reasoning = df["instructed_reasoning_type"][i]
        this_correct = df["correct"][i]
        model_input = df["model_input"][i]
            
        if positive_negative_dict.get(this_id) is None:
            positive_negative_dict[this_id] = {
                "question": model_input,
                "matching": [],
                "not_matching": []
            }

        example = {
            "model_output": this_output,
            "instructed_reasoning_type": this_instructed_reasoning,
            "correct": this_correct,
        }
        if compliance:
            positive_negative_dict[this_id]["matching"].append(example)
        else:
            example["gemini_reasoning_type"] = this_actual_reasoning
            positive_negative_dict[this_id]["not_matching"].append(example)
    

    # use all pairs
    question_set = set()
    for question in positive_negative_dict:
        for pos in positive_negative_dict[question]["matching"]:
            for neg in positive_negative_dict[question]["not_matching"]:
                if pos["instructed_reasoning_type"] != neg["instructed_reasoning_type"]:
                    continue
                if pos["correct"] != neg["correct"]:
                    continue

                question_set.add(positive_negative_dict[question]["question"])
                dataset[dataset_name].append({
                    "question": positive_negative_dict[question]["question"],
                    "matching": pos["model_output"],
                    "not_matching": neg["model_output"]
                })

    print(len(dataset[dataset_name]))
    print("Unique question number:", len(question_set))
    with open(f"steer_data/steering_data_all_pairs_{args.model_name}_{args.dataset}_compliance_same_labels.json", "w") as f:
        json.dump(dataset, f, indent=4)

