import os
import json
import re
from pathlib import Path

def extract_passk_results(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()


    pattern = r"\{.*pass@1.*\}"

    for line in reversed(lines):  
        match = re.search(pattern, line)
        if match:
            result_str = match.group(0)


            cleaned = re.sub(r"np\.float64\((.*?)\)", r"\1", result_str)
            cleaned = cleaned.replace("'", '"')  

            try:
                return json.loads(cleaned)
            except Exception as e:
                print(f" {log_path} -> {e}")
                print(f" {cleaned}")
    return None

def process_all_logs(root_dir, output_jsonl="passk_summary.jsonl"):
    root_dir = Path(root_dir)
    output = []

    for log_file in root_dir.rglob("*.log"):
        result = extract_passk_results(log_file)
        if result:
            log_parent = log_file.parent.name
            print(f" {log_file}")
            output.append({
                "folder": log_parent,
                "results": result
            })
        else:
            print(f" {log_file}")


    with open(output_jsonl, 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')

    print(f"Extracted results from {len(output)} log files to {output_jsonl}")


process_all_logs("")