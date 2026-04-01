import re
from datasets import Dataset, DatasetDict, load_dataset
import json
import pandas as pd
from argparse import ArgumentParser
from os.path import isfile, join
from os import listdir
import matplotlib.pyplot as plt
from induce_reasoning import get_dataset, is_correct, correct_format


def get_accuracy(file_path: str, result_dir: str):
    # dataset_name = file_path.split("/")[-1].replace("_results.json", "")
    dataset_name = file_path.split("/")[-1].split("_")[:2]
    dataset_name = "_".join(dataset_name)

    question_list, gold_answer_list, reasoning_types, id_list = get_dataset(dataset_name)

    with open(file_path, "r") as f:
        response_list = json.load(f)
    
    output_file = []
    correct_list = []
    for i in range(len(response_list)):
        response = response_list[i]["pred"][0]
        gold_answer = gold_answer_list[i]

        correctness = is_correct(response.lower(), gold_answer, dataset_name)
        output_file.append({
            "id": id_list[i],
            "prompt": question_list[i],
            "gold_answer": gold_answer,
            "response": response,
            "reasoning_type": reasoning_types[i],
            "correct": correctness
        })
        correct_list.append(correctness)

    acc = sum(correct_list)/len(correct_list)
    print(f"Accuracy: {acc}")
    save_name = "_".join(file_path.split("/")[1:])
    with open(join(result_dir, save_name), "w") as f:
        json.dump(output_file, f, indent=4)

    return acc

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--steered_dir", type=str, default="steered_outputs")
    parser.add_argument("--dir", type=str, default="output_files/")
    args = parser.parse_args()

    check_list = list(listdir(args.steered_dir))
    print(len(check_list))

    df_data = []
    for file in check_list:
        if file == "backup":
            continue
        if not isfile(join(args.steered_dir, file)):
            layer_name = file.split("+")[0]
            layer_name = layer_name.split("_")[-1]
            layer_name = int(layer_name)

            model_name = file.split("_")[0]

            for result_file in listdir(join(args.steered_dir, file)):
                mu = result_file.split("mu")[-1].split("_")[0]
                mu = float(mu)
                reasoning_type = result_file.split("_results")[0].split("_")[-1]
                if result_file.endswith("_results.json"):
                    file_path = join(args.steered_dir, file, result_file)
                    print(f"Evaluating {file_path}...")
                    acc = get_accuracy(file_path, args.dir)
                    df_data.append({
                        "model": model_name,
                        "layer": layer_name,
                        "mu": mu,
                        "instructed_reasoning_type": reasoning_type,
                        "accuracy": acc
                    })
    df = pd.DataFrame(df_data)
    print(df)
    df.to_csv("steering_layer_accuracies.csv", index=False)
