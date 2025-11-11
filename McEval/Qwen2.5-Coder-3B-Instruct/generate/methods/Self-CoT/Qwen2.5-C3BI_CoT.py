import os
import re
import yaml
import json
import random
import logging
import argparse
import tempfile
import torch
import shutil
import subprocess
import textwrap
from typing import List, Dict, Any, Tuple, Optional
from multiprocessing import set_start_method, Process
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# Set GPU
# ----------------------------------------------------------------------------

# Physical GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_gpu_memory_usage() -> List[float]:
    """
    Calculate the CUDA_VISIBLE_DEVICES's memory usage fraction
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        memory_info = result.strip().split("\n")    # used_memory,total_memory
    except subprocess.CalledProcessError:
        # Fallback if nvidia-smi fails
        logging.error("nvidia-smi failed. Falling back to CUDA_VISIBLE_DEVICES.")
        print("nvidia-smi failed. Falling back to CUDA_VISIBLE_DEVICES.")

        # Return dummy memory usage for all visible devices
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices:
            return [0.0] * len(visible_devices.split(","))
        else:
            return []

    usage_fractions = []
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        visible_indices = list(map(int, visible_devices.split(",")))
    else:
        logging.error("CUDA_VISIBLE_DEVICES is not set. Exiting program.")
        print("CUDA_VISIBLE_DEVICES is not set. Exiting program.")
        exit(1)

    physical_to_logical = {phys_id: log_id for log_id, phys_id in enumerate(visible_indices)}
    logging.info(f"Physical to logical GPU mapping: {physical_to_logical}")
    print(f"Physical to logical GPU mapping: {physical_to_logical}")

    for i, info in enumerate(memory_info): # Calculate the memory usage fraction
        if i in visible_indices:
            used, total = map(int, info.split(","))
            usage_fractions.append(used / total)

    return usage_fractions


def get_gpus_below_threshold(threshold: float = 0.8) -> List[int]:
    """
    Retrieve a list of GPU IDs with memory usage below a specified threshold.
    """
    usage_fractions = get_gpu_memory_usage()
    available_gpus = [i for i, usage in enumerate(usage_fractions) if usage < threshold]

    for i, usage in enumerate(usage_fractions):
        logging.info(f"GPU {i}: {usage * 100:.2f}% used")
        print(f"GPU {i}: {usage * 100:.2f}% used")

    if not available_gpus:
        logging.warning(f"No GPUs found with memory usage below {threshold * 100}%.")
        print(f"No GPUs found with memory usage below {threshold * 100}%.")
    return available_gpus

def set_device(threshold: float = 0.8) -> List[Tuple[int, torch.device]]:
    """
    Set the device(s), prioritizing GPUs with memory usage below the specified threshold.
    Returns a list of (physical_id, torch.device) tuples.
    """
    if torch.cuda.is_available():
        usage_fractions = get_gpu_memory_usage()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        physical_to_logical = {phys_id: log_id for log_id, phys_id in enumerate(map(int, visible_devices))}
        available_gpus = [
            (phys_id, torch.device(f"cuda:{logical_id}"))
            for phys_id, logical_id in physical_to_logical.items()
            if usage_fractions[logical_id] < threshold
        ]

        if available_gpus:
            logging.info(
                f"Using GPUs: {[(phys, dev.index) for phys, dev in available_gpus]} with memory usage below {threshold * 100}%."
            )
            print(
                f"Using GPUs: {[(phys, dev.index) for phys, dev in available_gpus]} with memory usage below {threshold * 100}%."
            )
            return available_gpus  
        else:
            logging.error("No GPUs meet the threshold requirements. Terminating program.")
            print("No GPUs meet the threshold requirements. Terminating program.")
            exit(1)
    else:
        logging.error("No GPU detected. Terminating program.")
        print("No GPU detected. Terminating program.")
        exit(1)

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
    
def load_model_and_tokenizer(config: Dict[str, Any], device: torch.device) -> Tuple[Any, Any]:
    """
    Load the specified model and tokenizer, and move the model to the specified device.
    """
    model_path = config["model"]["model_path"]
    logging.info(f"Loading model from: {model_path}")

    # Load the specified model and tokenizer, supporting both `pytorch_model.bin` and `safetensors` formats.
    # Dynamically handle shard count based on the YAML configuration.
    model_path = config["model"]["model_path"]
    safetensors_shards = config["model"].get("safetensors_shards", 0)
    if safetensors_shards <= 0:
        logging.warning("No safetensors shards configured, proceeding without shard loading.")
    logging.info(f"Loading model from: {model_path}")

    # Prioritize checking for PyTorch format files
    pytorch_model_file = os.path.join(model_path, "pytorch_model.bin")
    safetensors_index_file = os.path.join(model_path, "model.safetensors.index.json")
    safetensors_files = []

    if os.path.exists(pytorch_model_file):
        logging.info("Detected `pytorch_model.bin`. Loading model in PyTorch format.")
        use_safetensors = False
    elif os.path.exists(safetensors_index_file):
        if safetensors_shards > 0:
            safetensors_files = [
                os.path.join(model_path, f"model-{i:05d}-of-{safetensors_shards:05d}.safetensors")
                for i in range(1, safetensors_shards + 1)
            ]
            if all(os.path.exists(file) for file in safetensors_files):
                logging.info(f"Detected safetensors files with {safetensors_shards} shards.")
                use_safetensors = True
            else:
                raise FileNotFoundError(
                    f"Some safetensors files are missing. Please check paths: {safetensors_files}"
                )
        else:
            logging.info("Detected `model.safetensors.index.json`. Enabling automatic sharded loading.")
            use_safetensors = True
    else:
        raise FileNotFoundError(
            f"No model files found (`pytorch_model.bin` or safetensors). Check path: {model_path}"
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Load model based on file format
    if not use_safetensors:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
        )

    # Set `pad_token` if not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set `pad_token` to `eos_token`.")

    # Move model to the specified device
    model = model.to(device)
    print(f"Model loaded and moved to device: {device}")
    logging.info(f"Model loaded and moved to device: {device}")

    return model, tokenizer

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

def filter_tasks(
    tasks: List[Dict[str, Any]], limit: Optional[int] = None, num_gpus: int = 1
) -> List[Dict[int, List[Dict[str, Any]]]]:
    """
    Filter and distribute tasks evenly across multiple GPUs.
    """

    if limit is not None:
        tasks = tasks[:limit]
        logging.info(f"Filtered tasks: limited to {len(tasks)} tasks.")
    else:
        logging.info(f"No limit set. Processing all {len(tasks)} tasks.")

    # Split tasks evenly across GPUs
    chunk_size = (len(tasks) + num_gpus - 1) // num_gpus  
    distributed_tasks = [
        tasks[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)
    ]

    # Create the GPU-task mapping
    gpu_task_mapping = [{gpu_id: gpu_tasks} for gpu_id, gpu_tasks in enumerate(distributed_tasks)]

    return gpu_task_mapping

# ----------------------------------------------------------------------------
# Generate xcot
# ----------------------------------------------------------------------------
def generate_xcot(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_length: int,
    temperature: float,
    top_p: float,
    n_samples: int,
    device: torch.device,
    config: Dict[str, Any],  
) -> List[str]:
    """
    Generate a xcot based on the provided prompt.
    """
    
    messages = [
        {
            "role": "system",
            "content": config["messages"]["xcot_system"],
        }
    ]

    for example in config["messages"]["xcot_examples"]:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})

    xcot_user_content = config["messages"]["xcot_user"].format(prompt=prompt)
    messages.append({"role": "user", "content": xcot_user_content})


    # Format input using chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    xcot_outputs = []
    for output in outputs:
        generated_tokens = output[input_length:]  
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        xcot_outputs.append(generated_text)
    return xcot_outputs

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

# ----------------------------------------------------------------------------
# Single gpu processing tasks
# ----------------------------------------------------------------------------
def process_tasks(
    model: Any,
    tokenizer: Any,
    tasks: List[Dict[str, Any]],
    max_length: int,
    temperature: float,
    top_p: float,
    n_samples: int,
    response_output_file: str,
    final_code_output_file: str,
    device: torch.device,
    config: Dict[str, Any],  
):

    """
    Process tasks to generate xcot and then generate the final code.
    """
    initialize_file(response_output_file)
    initialize_file(final_code_output_file)

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        entry_point = task["entry_point"]

        logging.info(f"Processing task: {task_id}")
        outputs = generate_code(
            model, tokenizer, entry_point, prompt, max_length, temperature, top_p, n_samples, device, config
        )
        
        # Function to extract Python code blocks from the output
        def extract_code(output: str) -> str:
            matches = re.findall(r"```python\n(.*?)\n```", output, re.DOTALL)  # Match code within ```python blocks
            return max(matches, key=len).strip() if matches else output.strip()  # Return the longest match or raw output

        
        if isinstance(outputs, list) and len(outputs) > 0:
            for single_output in outputs:
                # Save the extracted code with task metadata
                code_result = {
                    "task_id": task_id,
                    "completion": single_output
                }
                try:
                    with open(response_output_file, "a") as code_file:
                        code_file.write(json.dumps(code_result, ensure_ascii=False) + "\n")
                except Exception as e:
                    logging.error(f"Error writing response result for task {task_id}: {e}")
                logging.info(f"Completed task {task_id}, Response saved.")  # Log progress


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

        
        

def process_tasks_for_gpu(
    gpu_id: int,
    model,
    tokenizer,
    tasks: List[Dict[str, Any]],
    max_length: int,
    temperature: float,
    top_p: float,
    n_samples: int,
    response_gpu_temp_file: str,
    final_code_temp_file: str,
    device: str,
    config: Dict[str, Any], 
):
    """
    Processes tasks for a single GPU and writes the results to a temporary file.
    """
    try:
        logging.info(f"Processing tasks on GPU {gpu_id} with device {device}")

        process_tasks(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
            response_output_file= response_gpu_temp_file,
            final_code_output_file=final_code_temp_file,
            device=device,
            config=config,
        )

        logging.info(f"GPU {gpu_id} tasks completed. Results saved to {final_code_temp_file}")

    except Exception as e:
        logging.error(f"Error processing tasks on GPU {gpu_id}: {e}")


# ----------------------------------------------------------------------------
# Merge the temp results and clean temp files.
# ----------------------------------------------------------------------------
def merge_temp_files(temp_files: List[str], final_output_file: str):
    """
     Merge multiple temporary files into a single final output file.
    """

    try:
        logging.info(f"Merging temporary files into final output file: {final_output_file}")

        final_results = []
        for temp_file in temp_files:
            with open(temp_file, "r") as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())  
                        final_results.append(result)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error parsing JSON from {temp_file}: {e}")
                        continue

        with open(final_output_file, "w") as f:
            for result in final_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logging.info(f"Final results saved to {final_output_file}")

    except Exception as e:
        logging.error(f"Error merging temporary files: {e}")


def cleanup_temp_files(temp_files: List[str]):
    """
    Clean up temporary files and the folders in which they reside.
    """
    try:
        logging.info("Cleaning up temporary files and their directory.")
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logging.info(f"Removed temporary file: {temp_file}")
        
        if temp_files:
            common_dir = os.path.dirname(temp_files[0])
            if os.path.exists(common_dir) and not os.listdir(common_dir):
                os.rmdir(common_dir)
                logging.info(f"Removed empty directory: {common_dir}")
        
        logging.info("Temporary files and directory cleaned up.")
    except Exception as e:
        logging.error(f"Error cleaning up temporary files and directory: {e}")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(config_path: str):
    # Load Config & Setup
    config = load_config(config_path)
    set_seed(config["seed"])
    setup_logging(config["logging"]["log_file"])

    devices = set_device()
    num_gpus = len(devices)
    set_seed(config["seed"])

    # Load Models & Data
    models_and_tokenizers = []
    for physical_id, device in devices: 
        model, tokenizer = load_model_and_tokenizer(config, device)  
        models_and_tokenizers.append((model, tokenizer))
        logging.info(f"Loading model on physical GPU: {physical_id} (logical: {device})")
        print(f"Model loaded on physical GPU: {physical_id} (logical: {device})")

    dataset = load_humaneval_dataset(config["dataset"]["dataset_file"])
    gpu_task_mapping = filter_tasks(dataset, limit=config.get("limit"), num_gpus=num_gpus)
    
    for gpu_mapping in gpu_task_mapping:
        for gpu_id, tasks in gpu_mapping.items():
            physical_id = next((phys_id for phys_id, dev in devices if dev.index == gpu_id), None)
            if tasks:  
                logging.info(f"Logical GPU {gpu_id} (Physical GPU {physical_id}): First task: {tasks[0]}")
                logging.info(f"Logical GPU {gpu_id} (Physical GPU {physical_id}): Last task: {tasks[-1]}")
            else:
                logging.warning(f"Logical GPU {gpu_id} (Physical GPU {physical_id}): No tasks assigned.")   

    # Create Temporary Files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir_name = "temp_files"
    temp_dir = os.path.join(current_dir, temp_dir_name)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"Temporary directory created: {temp_dir}")
        logging.info(f"Temporary directory created: {temp_dir}")
    else:
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        print(f"Temporary directory created: {temp_dir}")
        logging.info(f"Temporary directory created: {temp_dir}")

    response_gpu_temp_files = [os.path.join(temp_dir, f"temp_response_gpu{gpu_id}.jsonl") for gpu_id in range(num_gpus)]
    for gpu_id, temp_file in enumerate(response_gpu_temp_files):
        physical_id = next((phys_id for phys_id, dev in devices if dev.index == gpu_id), None)
        logging.info(
            f"Response temporary file created for GPU (Physical: {physical_id}, Logical: {gpu_id}): {temp_file}"
    )

    final_code_temp_files = [os.path.join(temp_dir, f"temp_final_code_gpu{gpu_id}.jsonl") for gpu_id in range(num_gpus)]
    for gpu_id, temp_file in enumerate(final_code_temp_files):
        physical_id = next((phys_id for phys_id, dev in devices if dev.index == gpu_id), None)
        logging.info(
            f"Final code temporary file created for GPU (Physical: {physical_id}, Logical: {gpu_id}): {temp_file}"
        )

    # Process Tasks (Multi-GPU)
    processes = []
    for gpu_id, (model, tokenizer) in enumerate(models_and_tokenizers):
        tasks = gpu_task_mapping[gpu_id][gpu_id]  
        response_gpu_temp_file = response_gpu_temp_files[gpu_id]
        final_code_temp_file = final_code_temp_files[gpu_id]

        p = Process(
            target=process_tasks_for_gpu,
            args=(
                gpu_id,
                model,
                tokenizer,
                tasks,
                config["model"]["max_length"],
                config["model"]["temperature"],
                config["model"]["top_p"],
                config["n_samples"],
                response_gpu_temp_file,
                final_code_temp_file,
                gpu_id,
                config,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Merge & Cleanup Temporary Files
    merge_temp_files(final_code_temp_files, config["generation"]["final_code_output_file"])
    merge_temp_files(response_gpu_temp_files, config["generation"]["response_output_file"])
    cleanup_temp_files(final_code_temp_files)
    cleanup_temp_files(response_gpu_temp_files)

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
