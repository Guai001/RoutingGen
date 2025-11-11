# RoutingGen-Code

This repository contains the official implementation and evaluation code of the paper "Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation". 


---

##  Directory Structure

| Directory           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `datasets/`         | Contains benchmark datasets used in the experiments.                        |
| `HumanEval/`        | Generation and evaluation code for the HumanEval benchmark.                 |
| `HumanEval_ET/`     | Evaluation code for the HumanEval-ET benchmark.                             |
| `MBPP/`             | Generation and evaluation code for the MBPP benchmark.                      |
| `MBPP_ET/`          | Evaluation code for the MBPP-ET benchmark.                                  |
| `McEval/`           | Generation and evaluation code for the McEval benchmark.                    |
| `OpenEval/`         | Generation and evaluation code for the OpenEval benchmark.                  |
| `routing/`          | Source code for difficulty-aware dynamic routing.                           |
| `diff_pass/`        | Source code to compute pass@k scores based on task difficulty.              |
| `token_consume/`    | Source code to calculate token usage during generation.                     |
| `requirements.txt`  | Required Python packages for installation.                                  |
| `README.md`         | This documentation file.                                                    |

## Supported Models

This repository supports three high-performing code generation models: Qwen2.5-Coder-3B-Instruct, DeepSeek-Coder-6.7B-Instruct, and DeepSeek-V3-671B. To use other models, simply modify the model path in the corresponding configuration files.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Generation

Example: Run RoutingGen on HumanEval using Qwen2.5-Coder-3B-Instruct:
```bash
cd RoutingGen-Code/HumanEval/Qwen2.5-Coder-3B-Instruct/generate/methods/RoutingGen

python Qwen2.5-C3BI_RG.py --config Qwen2.5-C3BI_RG.yaml
```
Replace with the appropriate script and config for other benchmarks and models.

## Evaluation

Example: Evaluate RoutingGen on HumanEval:
```bash
cd RoutingGen-Code/HumanEval/Qwen2.5-Coder-3B-Instruct/evaluate/human_eval

python evaluate_functional_correctness.py \
  --config_file=../../evaluate/run_yaml/run_eval.yaml \
  > ../../evaluate/results/RoutingGen/RG.log 2>&1
```
Replace with corresponding paths for other benchmarks and methods.



