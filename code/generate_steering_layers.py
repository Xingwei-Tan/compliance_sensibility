import yaml
from argparse import ArgumentParser
import sys
sys.path.append('../')
from omegaconf import OmegaConf, DictConfig
from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.datasets import prepare_train_dataset
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.datasets import prepare_generation_datasets
import json
import pandas as pd
from os.path import isfile
from induce_reasoning import get_dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="The layer to steer in the model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--data_path", type=str, default="steer_data/steering_data_all_pairs_Llama-3.3-70B-Instruct_anli_train_abductive.json")
    args = parser.parse_args()


    with open(args.data_path, "r") as f:
        data_dict = json.load(f)

    subset_name = list(data_dict.keys())[0]
    simple_model_name = args.model_name.split("/")[-1]


    generate_caa_arguments = {
        "alg_name": "caa",
        "layers": [args.layer],
        "save_activations": True,
        "multiple_choice": False
    }

    with open(f'configs/generate_caa_{args.layer}.yaml', 'w') as file:
        yaml.dump(generate_caa_arguments, file, default_flow_style=False, sort_keys=False)

        
    control_arguments = {
        "model_name_or_path": args.model_name,
        "torch_dtype": "bfloat16",
        "device": "auto",
        "use_chat_template": True,
        "system_prompt": "",
        "steer_train_hparam_paths": [f"configs/generate_caa_{args.layer}.yaml"],
        "steer_train_dataset": [subset_name],
        "steer_vector_output_dirs": f"compliance_vectors_same_labels/{subset_name}_{args.layer}",
        "steer_vector_load_dir": [f"compliance_vectors_same_labels/{subset_name}_{args.layer}/{subset_name}/caa_vector"],
        "generation_data_size": -1,
        "generation_output_dir": f"compliance_steered_outputs/{subset_name}_{args.layer}/",
        "num_responses": 1,
        "steer_from_end_position": False,
        "generation_params": {
            "max_new_tokens": 2048,
            "temperature": 0.1,
        }
    }

    with open(f'configs/control_{simple_model_name}_{subset_name}_layer_{args.layer}.yaml', 'w') as file:
        yaml.dump(control_arguments, file, default_flow_style=False, sort_keys=False)
        
    top_cfg = OmegaConf.load(f'configs/control_{simple_model_name}_{subset_name}_layer_{args.layer}.yaml')

    if not isfile(f"compliance_vectors_same_labels/{subset_name}_{args.layer}/{subset_name}/caa_vector/layer_{args.layer}.pt"):
        vector_generator = BaseVectorGenerator(top_cfg)
        _ = vector_generator.generate_vectors(data_dict)

