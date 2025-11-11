import json
import os
import numpy as np
import itertools
import pandas as pd
from typing import List, Union
from collections import defaultdict


def load_easy_tasks(difficulty_file: str) -> List[str]:
    easy_tasks = []
    with open(difficulty_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("routing_label") == "Easy":
                    easy_tasks.append(data["task_id"])
            except Exception:
                continue

    return easy_tasks


def load_all_task_ids(difficulty_file: str) -> List[str]:
    task_ids = set()
    with open(difficulty_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                task_ids.add(data["task_id"])
            except:
                continue
    return list(task_ids)


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def collect_task_correct_counts(folder_path: str, easy_tasks: List[str]):
    simple_total_counts = defaultdict(int)
    simple_correct_counts = defaultdict(int)
    complex_total_counts = defaultdict(int)
    complex_correct_counts = defaultdict(int)


    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jsonl') or file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            task_id = data.get("task_id")
                            passed = data.get("passed", False)
                            if not task_id:
                                continue
                            if task_id in easy_tasks:
                                simple_total_counts[task_id] += 1
                                if passed:
                                    simple_correct_counts[task_id] += 1
                            else:
                                complex_total_counts[task_id] += 1
                                if passed:
                                    complex_correct_counts[task_id] += 1
                        except Exception:
                            continue

    return simple_total_counts, simple_correct_counts, complex_total_counts, complex_correct_counts


def process_one_folder(folder_path: str, easy_tasks: List[str], n: int = 20):
    simple_total_counts, simple_correct_counts, complex_total_counts, complex_correct_counts = collect_task_correct_counts(folder_path, easy_tasks)

    simple_passk = {k: 0.0 for k in [1, 3, 5]}
    complex_passk = {k: 0.0 for k in [1, 3, 5]}


    
    if len(simple_total_counts):
        n_list = list(simple_total_counts.values())
        c_list = [simple_correct_counts[tid] for tid in simple_total_counts]

        for k in [1, 3, 5]:
            p = estimate_pass_at_k(n_list, c_list, k).mean()
            simple_passk[k] = f"{round(p * 100, 2)}%"

    if len(complex_total_counts):
        n_list = list(complex_total_counts.values())
        c_list = [complex_correct_counts[tid] for tid in complex_total_counts]
        for k in [1, 3, 5]:
            p = estimate_pass_at_k(n_list, c_list, k).mean()
            complex_passk[k] = f"{round(p * 100, 2)}%"

    return {
        "Simple": [simple_passk[1], simple_passk[3], simple_passk[5]],
        "Complex": [complex_passk[1], complex_passk[3], complex_passk[5]]
    }


def save_all_results_to_excel(
    results_root_list: List[str],
    easy_tasks: List[str],
    total_task_ids: List[str],
    output_excel: str,
    n: int = 20
):
    all_records = []

    simple_count = len(easy_tasks)
    total_count = len(total_task_ids)
    complex_count = total_count - simple_count

    for results_root in results_root_list:
       
        parts = results_root.strip(os.sep).split(os.sep)
        group_name = "/" + "/".join(parts[-4:-2])
        all_records.append({
            "Folder": group_name,
            "pass@1 (Simple)": "",
            "pass@1 (Complex)": "",
        })


        all_records.append({
            "Folder": "Sample Count",
            "pass@1 (Simple)": simple_count,
            "pass@1 (Complex)": complex_count,
        })

        for subfolder in sorted(os.listdir(results_root)):
            subfolder_path = os.path.join(results_root, subfolder)
            if os.path.isdir(subfolder_path):
                result = process_one_folder(subfolder_path, easy_tasks, n)
                all_records.append({
                    "Folder": subfolder,
                    "pass@1 (Simple)": result["Simple"][0],
                    "pass@1 (Complex)": result["Complex"][0],
                })

    df = pd.DataFrame(all_records)
    df.to_excel(output_excel, index=False)


def main():
    difficulty_file = ""

   
    results_root_list = [
        "",
        "",
        ""
    ]

    output_excel = "diff_pass_all_results.xlsx"

    easy_tasks = load_easy_tasks(difficulty_file)
    total_task_ids = load_all_task_ids(difficulty_file)
    save_all_results_to_excel(results_root_list, easy_tasks, total_task_ids, output_excel, n=20)


if __name__ == "__main__":
    main()