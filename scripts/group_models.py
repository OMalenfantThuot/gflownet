#! /usr/bin/env python

import os
import shutil
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", default="inference")
args = parser.parse_args()

paramdirs = [folder for folder in os.listdir() if folder.startswith("T_")]

os.makedirs(args.savedir, exist_ok=True)
final_model_name = "final_model.torch"

for paramdir in paramdirs:
    try:
        rundirs = os.listdir(paramdir)
        for rundir in rundirs:
            saved_models = [
                f
                for f in os.listdir(os.path.join(paramdir, rundir))
                if f.endswith(".torch")
            ]
            if final_model_name in saved_models:
                best_model_name = final_model_name
            else:
                checkpoint_values = [
                    int(name.split("_")[1].split(".")[0]) for name in saved_models
                ]
                last_value = np.argmax(checkpoint_values)
                best_model_name = saved_models[last_value]

            shutil.copyfile(
                os.path.join(paramdir, rundir, best_model_name),
                os.path.join("inference", f"{paramdir}_{rundir}_{best_model_name}"),
            )
    except FileNotFoundError:
        pass
