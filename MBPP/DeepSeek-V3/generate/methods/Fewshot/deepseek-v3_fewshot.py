import os
import re
import yaml
import json
import random
import logging
import time
import argparse
import tempfile
import torch
import requests
import shutil
import subprocess
import tiktoken
import textwrap
from typing import List, Dict, Any, Tuple, Optional
from multiprocessing import set_start_method, Process
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# Set GPU
# ----------------------------------------------------------------------------

# Physical GPU
os.environ["CUDA_VISIBLE_DEVICES"] = " "


# ----------------------------------------------------------------------------
# Initialize file and seed
# ----------------------------------------------------------------------------
def initialize_file(file_path: str):
    """
     Initialize a file by clearing its contents.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.truncate(0)

def setup_logging(log_file: str):
    """
    Configure logging and clear the log file at startup.
    Logs will be written only to the specified file.
    """
    initialize_file(log_file)  

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  

    file_handler = logging.FileHandler(log_file, mode="a") 
    file_handler.setLevel(logging.DEBUG) 
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  
    file_handler.setFormatter(file_formatter)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.addHandler(file_handler)

    logger.info("Log initialization completed.") 

def set_seed(seed: int):
    """
    Set the random seed to ensure reproducibility.
    """
    random.seed(seed)
    logging.info(f"Python random seed set to {seed}")
    torch.manual_seed(seed)
    logging.info(f"PyTorch global random seed (CPU) set to {seed}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            with torch.cuda.device(gpu_id):
                logging.info(f"Random seed {seed} set for GPU {gpu_id}")
                print(f"Random seed {seed} set for GPU {gpu_id}")
    else:
        logging.info("No CUDA device available. Skipping GPU random seed setting.")

# ----------------------------------------------------------------------------
# Load model and tokenizer
# ----------------------------------------------------------------------------
def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load the YAML configuration file.
    """
    with open(config_file, "r") as file:
        return yaml.safe_load(file)
    

# ----------------------------------------------------------------------------
# Load datasets and filter tasks
# ----------------------------------------------------------------------------
def load_humaneval_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """
    Load the HumanEval dataset.
    """
    with open(dataset_file, "r") as file:
        dataset = [json.loads(line) for line in file]

    logging.info(f"Loaded HumanEval dataset with {len(dataset)} tasks.")
    return dataset


# ----------------------------------------------------------------------------
# Generate Code Based on Prompt and xcot
# ----------------------------------------------------------------------------
def generate_code(
    model: Any,
    tokenizer: Any,
    entry_point: str,
    prompt: str,
    max_length: int,
    temperature: float,
    top_p: float,
    n_samples: int,
    device: torch.device,
    config: Dict[str, Any],  
) -> List[str]:
    # model, tokenizer, entry_point, prompt, max_length, temperature, top_p, n_samples, device, config
    
    do_sample = temperature is not None and temperature > 0
    if not do_sample:  
        top_p = None  
        n_samples = 1  
        temperature = None  

    messages = [
        {
            "role": "system",
            "content": config["messages"]["code_system"],
        }
    ]

    code_user_content = config["messages"]["code_user"].format(prompt=prompt)
    messages.append({"role": "user", "content": code_user_content})

    # Apply chat template and prepare inputs
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    input_length = inputs["input_ids"].shape[1]

    if not do_sample:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_samples,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            top_k = None,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_samples,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    code_outputs = []
    for output in outputs:
        generated_tokens = output[input_length:]  
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        code_outputs.append(generated_text)
    
    return code_outputs

def generate_code_via_api(
    prompt: str,
    n_samples: int,
    config: Dict[str, Any],
    api_key: str,
    model_name: str ,
    base_url: str ,
    delay_between_requests: float ,
    max_length: int ,
    temperature: float ,
    top_p: float,
    do_sample: bool = True,
    test_case: str = " ",
) -> List[str]:

    do_sample = temperature is not None and temperature > 0
    if not do_sample:  
        top_p = None  
        n_samples = 1  
        temperature = None  

    messages = [
        {
            "role": "system",
            "content": config["messages"]["code_system"],
        }
    ]
    for example in config["messages"]["code_examples"]:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})

    code_user_content = config["messages"]["code_user"].format(prompt=prompt,test_case=test_case)
    messages.append({"role": "user", "content": code_user_content})

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n_samples,
        "max_tokens": max_length,
    })

    try:
        response = requests.request("POST",base_url, headers=headers, data=payload, timeout=60)
        # print("Raw response text:", response.text) 
        response.raise_for_status()
        result = response.json()
        # print(result)
        generations = [choice["message"]["content"] for choice in result.get("choices", [])]
        time.sleep(delay_between_requests)  
        return generations
    except Exception as e:
        logging.error(f"API request failed: {e}")
        return []

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["seed"])
    setup_logging(config["logging"]["log_file"])
    
    dataset = load_humaneval_dataset(config["dataset"]["dataset_file"])
    
    final_code_output_file = config["generation"]["final_code_output_file"]
    code_output_file = config["generation"]["code_output_file"]

    n_samples = config["n_samples"]
    

    initialize_file(final_code_output_file)


    for data in dataset:
        task_id = data["task_id"]
        prompt = data["prompt"]
        test_case = data["test_list"]

        outputs = generate_code_via_api(
            prompt=prompt,
            n_samples=n_samples,
            config=config,
            api_key=config["api"]["key"],
            model_name=config["api"]["model"],
            base_url=config["api"]["base_url"],
            delay_between_requests=config["api"]["delay_between_requests"],
            max_length=config["model"]["max_length"],
            temperature=config["model"]["temperature"],
            top_p=config["model"]["top_p"],
            do_sample=config["model"].get("do_sample", True),
            test_case=test_case
        )

        def extract_code(output: str) -> str:
            matches = re.findall(r"```python\n(.*?)\n```", output, re.DOTALL)  # Match code within ```python blocks
            return max(matches, key=len).strip() if matches else output.strip()  # Return the longest match or raw output

        if isinstance(outputs, list) and len(outputs) > 0:
            for single_output in outputs:
                code_result = {
                    "task_id": task_id,
                    "completion": single_output
                }
                try:
                    with open(code_output_file, "a") as code_file:
                        code_file.write(json.dumps(code_result, ensure_ascii=False) + "\n")
                except Exception as e:
                    logging.error(f"Error writing code-output result for task {task_id}: {e}")
                logging.info(f"Completed task {task_id}, Code-output saved.")  # Log progress


                cleaned_output = extract_code(single_output)  # Clean the output to extract Python code
                # Save the extracted code with task metadata
                code_result = {
                    "task_id": task_id,
                    "completion": cleaned_output
                }
                try:
                    with open(final_code_output_file, "a") as code_file:
                        code_file.write(json.dumps(code_result, ensure_ascii=False) + "\n")
                except Exception as e:
                    logging.error(f"Error writing code result for task {task_id}: {e}")
                logging.info(f"Completed task {task_id}, Code saved.")  # Log progress
            

if __name__ == "__main__":
    set_start_method("spawn")
    
    parser = argparse.ArgumentParser(description="Run model with YAML config")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    model_name = config["model"].get(
        "model_name", "unascertained model"
    )  
    parser.description = f"Run {model_name} with YAML config"
    main(args.config)
