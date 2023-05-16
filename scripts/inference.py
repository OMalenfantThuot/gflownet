#! /usr/bin/env python

import os
import subprocess
import numpy as np
import pickle
import argparse
from spingflow.modeling.utils import add_modeling_arguments_to_parser


parser = argparse.ArgumentParser(
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser = add_modeling_arguments_to_parser(parser)
parser.add_argument("--property", default="magn", help="Property to predict")
parser.add_argument("--output_name", default="results_dict.pkl", help="Output name")
args = parser.parse_args()

# Parameters
params = {
    "nsamples": 50000,
    "batch_size": 1000,
    "device": "cpu",
}

inference_script_path = "/home/olimt/projects/rrg-cotemich-ac/olimt/Gflow/gflownet/scripts/spin_gflow_run.py"
base_inference_command = f"python {inference_script_path}"

for k, v in vars(args).items():
    if isinstance(v, bool):
        if v:
            base_inference_command += f" --{k}"
    else:
        if k != "output_name":
            base_inference_command += f" --{k} {v}"

for k, v in params.items():
    base_inference_command += f" --{k} {v}"


all_models = [f for f in os.listdir() if f.endswith(".pth") or f.endswith(".torch")]
temperatures = sorted(set([int(model.split("_")[1]) for model in all_models]))

results_dict = {}
for temperature in temperatures:
    models = [model for model in all_models if model.startswith(f"T_{temperature}_")]
    results_dict[f"T_{temperature}"] = np.empty(len(models))
    for i, model in enumerate(models):
        print(f"Predicting model : {model}")
        inference_command = base_inference_command + f" --savepath {model}"
        out = float(subprocess.check_output(inference_command, shell=True))
        results_dict[f"T_{temperature}"][i] = out

with open(args.output_name, "wb") as f:
    pickle.dump(results_dict, f)
