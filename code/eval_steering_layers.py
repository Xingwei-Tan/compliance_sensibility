import yaml
from argparse import ArgumentParser
import sys
from omegaconf import OmegaConf, DictConfig
from steer.vector_appliers.vector_applier import BaseVectorApplier
import json
import pandas as pd
from os.path import isfile
from induce_reasoning import get_dataset
import random
import torch
import numpy as np


INDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing inductive reasoning. You are given a set of observations. Your task is to infer the most probable general rule that explains all observations. Then, use the inferred rule to make predictions about new observations. Provide a detailed reasoning process leading to your conclusion."""

DEDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing deductive reasoning. You are given a general rule and specific observations. Your task is to apply the general rule to the observations to derive a logically certain conclusion. Provide a detailed reasoning process leading to your conclusion."""

ABDUCTIVE_PROMPT = """Instruction: You are a logician tasked with performing abductive reasoning. You are given a general rule and an observation. Your task is to generate the most probable and simplest hypothesis that, if true, would logically explain all the observations provided. Provide a detailed reasoning process leading to your conclusion."""

PROMPT_DICT = {
    "inductive": INDUCTIVE_PROMPT,
    "deductive": DEDUCTIVE_PROMPT,
    "abductive": ABDUCTIVE_PROMPT
}


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_instruction(instruction: str, dataset_dict):
    key = list(dataset_dict.keys())[0]
    for i in range(len(dataset_dict[key])):
        dataset_dict[key][i]["input"] = instruction + "\n\n" + dataset_dict[key][i]["input"]
    return dataset_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="The layer to steer in the model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--eval_data", type=str, default="anli_test")
    parser.add_argument("--type", type=str, default="abductive", choices=["inductive", "deductive", "abductive"], help="The type of reasoning to evaluate.")
    parser.add_argument("--data_path", type=str, default="steer_data/steering_data_all_pairs_Llama-3.3-70B-Instruct_anli_train_abductive.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.data_path, "r") as f:
        data_dict = json.load(f)

    subset_name = list(data_dict.keys())[0]
    simple_model_name = args.model_name.split("/")[-1]

    question_list, _, _, _ = get_dataset(args.eval_data)
    eval_datasets = {args.eval_data: []}
    for i in range(len(question_list)):
        eval_datasets[args.eval_data].append({
            "input": question_list[i],
        })

    
    print(f"Loaded Test Data: {len(eval_datasets[args.eval_data])}")

      
    apply_caa_arguments = {
        "alg_name": "caa",
        "layers": [args.layer, args.layer+1, args.layer+2, args.layer+3], # you can customize the layers to apply CAA
        "save_activations": True
    }

    with open(f'configs/apply_caa_4_{args.layer}.yaml', 'w') as file:
        yaml.dump(apply_caa_arguments, file, default_flow_style=False, sort_keys=False)
        
        
    control_arguments = {
        "model_name_or_path": args.model_name,
        "torch_dtype": "bfloat16",
        "device": "auto",
        "use_chat_template": True,
        "system_prompt": "",
        "steer_vector_output_dirs": f"compliance_vectors_same_labels/{subset_name}_{args.layer}",
        "apply_steer_hparam_paths": [f"configs/apply_caa_4_{args.layer}.yaml"],
        "steer_vector_load_dir": [f"compliance_vectors_same_labels/{subset_name}_{args.layer}/{subset_name}/caa_vector"],
        "generation_data_size": -1,
        "generation_output_dir": f"steered_outputs_same_labels/{subset_name}_{args.layer}+4/",
        "num_responses": 1,
        "steer_from_end_position": False,
        "generation_params": {
            "max_new_tokens": 4096,
            "temperature": 0.5,
        }
    }

    with open(f'configs/control_{simple_model_name}_{subset_name}_layer_{args.layer}.yaml', 'w') as file:
        yaml.dump(control_arguments, file, default_flow_style=False, sort_keys=False)
        
    top_cfg = OmegaConf.load(f'configs/control_{simple_model_name}_{subset_name}_layer_{args.layer}.yaml')


    vector_applier = BaseVectorApplier(top_cfg)
    eval_datasets = add_instruction(PROMPT_DICT[args.type], eval_datasets)
    mu_range = [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for mu in mu_range: # You can customize your own multipliers
        vector_applier.hparams_dict["caa"].multipliers = [mu]
        vector_applier.apply_vectors()



        results = vector_applier.generate(eval_datasets, save_results=False)
        vector_applier.save_results(results, f"{args.eval_data}_{args.layer}+4_mu{mu}_{args.seed}_{args.type}")

        vector_applier.model.reset_all()

