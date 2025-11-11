import json
import numpy as np
import yaml
import sys
from tqdm import tqdm
import signal
import traceback
import multiprocessing
import argparse
import time

def load_task_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []

def estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k for multiple problems in batch.
    """
    def estimator(n: int, c: int, k: int):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])



def timeout_handler(signum, frame):
    raise TimeoutError("Task execution timed out.")

def evaluate_candidate(task_id, completion, test_imports, test_list, timeout=3):  
    exec_context = {}

    try:
      
        for imp in test_imports:
            exec(imp, exec_context)
    except ImportError as e:
        print(f"Import error in task {task_id}: {e}")
        return 0, len(test_list)  

    try:
      
        exec(completion, exec_context)
    except SyntaxError as e:
        print(f"Syntax error in task {task_id}: {e}")
        return 0, len(test_list)  
    except NameError as e:
        print(f"Name error in task {task_id}: {e}")
        return 0, len(test_list)  
    except Exception as e:
        print(f"Execution error in task {task_id}: {e}")
        return 0, len(test_list) 


    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    passed_tests = 0
    try:

        for test_case in test_list:
            try:
                exec(test_case, exec_context)
                passed_tests += 1
            except AssertionError:
                pass  
            except NameError as e:
                print(f"Test case error in task {task_id}: {e}")
            except Exception as e:
                print(f"Test case error in task {task_id}: {e}")
    except TimeoutError:
        print(f"Task {task_id} timed out.")
        return 0, len(test_list)  
    finally:
        signal.alarm(0)  

    return passed_tests, len(test_list)


def run_evaluation(task_id, completion, test_imports, test_list, queue):
    try:
        result = evaluate_candidate(task_id, completion, test_imports, test_list)
        queue.put((task_id, result))
    except Exception as e:
        print(f"Error evaluating task_id {task_id}: {e}")
        queue.put((task_id, (0, len(test_list))))

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file) 
    except yaml.YAMLError as e:
        print(f"YAML parsing error in file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading config file {config_path}: {e}")
        sys.exit(1)

def main():
    

    parser = argparse.ArgumentParser(description="Evaluate code completions.")
    parser.add_argument('--config', required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()


    config = load_config(args.config)


    dataset_file = config["dataset_file"]
    generate_file = config["generate_file"]
    output_file = config["output_file"]
    ks = list(map(int, config["k"].split(',')))  
    timeout = config["timeout"]


    log_file = output_file.replace(".jsonl", ".log")
    sys.stdout = open(log_file, "w")


    print("Reading samples...\n")
    dataset_data = load_task_data(dataset_file)
    generate_data = load_task_data(generate_file)


    if not dataset_data or not generate_data:
        print("Error: Dataset or generated data is empty. Please check the input files.")
        return

    dataset_map = {task['task_id']: task for task in dataset_data}    

    results = []
    task_results = {}  
    total_tasks = len(generate_data)


    print("Running test suites...\n")
    progress_bar = tqdm(total=total_tasks, desc="Running test suites", unit="task", file=sys.stdout)
    try:
        with open(output_file, 'w') as output_f:
            for index, generate_task in enumerate(generate_data):
                task_id = generate_task.get('task_id')
                completion = generate_task.get('completion')
    
                if task_id not in dataset_map:
                    progress_bar.update(1)
                    continue
                

                task_data = dataset_map[task_id]
                test_imports = task_data.get("test_imports", [])
                test_list = task_data.get("test_list", [])

                if not test_list:
                    print(f"Task {task_id} has no test cases.")
                    progress_bar.update(1)
                    continue
                


                queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=run_evaluation,
                    args=(task_id, completion, test_imports, test_list, queue)
                )
                process.start()
                process.join(timeout=timeout)  

                if process.is_alive():
                    process.terminate()
                    process.join()
                    print(f"Task {task_id} timed out.")
                    passed_tests, total_tests = 0, len(test_list)
                else:
                    task_id, (passed_tests, total_tests) = queue.get()

              
                if task_id not in task_results:
                    task_results[task_id] = {"total": 0, "correct": 0}

                task_results[task_id]["total"] += 1
                if passed_tests == total_tests and total_tests > 0:
                    task_results[task_id]["correct"] += 1

          
                result_data = {
                    "task_id": task_id,
                    "completion": completion,
                    "result": "passed" if passed_tests == total_tests else "failed",
                    "passed": passed_tests == total_tests
                }
                results.append(result_data)

       
                output_f.write(json.dumps(result_data) + '\n')
                output_f.flush()

                progress_bar.update(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        progress_bar.close()

    

    total = np.array([task_results[task_id]["total"] for task_id in task_results])
    correct = np.array([task_results[task_id]["correct"] for task_id in task_results])


    pass_at_k_results = {}
    for k in ks:

        pass_at_k = estimate_pass_at_k(total, correct, k).mean()
        pass_at_k_results[f"pass@{k}"] = pass_at_k


    print("\nPass@k Results:", pass_at_k_results)



if __name__ == "__main__":
    main()