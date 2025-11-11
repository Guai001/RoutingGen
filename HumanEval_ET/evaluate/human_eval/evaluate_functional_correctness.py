import fire
import sys
import os
import yaml  
import json  


current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.abspath(os.path.join(current_dir, ".."))  
sys.path.append(project_root)  

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def load_config_from_yaml(yaml_file: str):
    """
    Load the YAML configuration file and return the parameters.
    """
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def prepare_output_file(output_file: str):
    """
    Prepare the output JSON file:
    - If the file does not exist, create it.
    - If the file exists, clear its content.
    """
    with open(output_file, "w") as file:
        file.write("")  # Clear the file content


def entry_point(config_file: str):
    """
    Evaluates the functional correctness of generated samples using parameters from a YAML configuration file.
    """
    # Load parameters from the YAML configuration file
    config = load_config_from_yaml(config_file)

    # Extract parameters from the configuration
    sample_file = config.get("sample_file")
    k = config.get("k", "1,5,10")
    n_workers = config.get("n_workers", 4)
    timeout = config.get("timeout", 3.0)
    problem_file = config.get("problem_file", HUMAN_EVAL)
    output_file = config.get("output_file", "output.json")  # Default output file name

    # Convert k to an integer list
    k = list(map(int, k.split(",")))

    # Prepare the output file (create or clear content)
    prepare_output_file(output_file)

    # Perform the evaluation
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file,output_file)

    # # Save the results to the output JSON file
    # with open(output_file, "w") as file:
    #     json.dump(results, file, indent=4)
    # print(f"Results have been saved to {output_file}")

    print(results)
    print(f"Results have been printed above")


def main():
    """
    Main function to invoke the script with Fire and pass the configuration file path.
    """
    fire.Fire(entry_point)


if __name__ == "__main__":
    sys.exit(main())
