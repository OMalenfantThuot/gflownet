#! /usr/bin/env python

import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("min_temp", type=int)
parser.add_argument("max_temp", type=int)
parser.add_argument("--step", type=int, default=1)
args = parser.parse_args()

temps = range(args.min_temp, args.max_temp + 1, args.step)

paramdirs = [f"T_{i}" for i in temps]
best_model_name = "best_model.pth"

os.makedirs("inference", exist_ok=True)

for paramdir in paramdirs:
    try:
        rundirs = os.listdir(paramdir)
        for rundir in rundirs:
            shutil.copyfile(
                os.path.join(paramdir, rundir, best_model_name),
                os.path.join("inference", f"{paramdir}_{rundir}_{best_model_name}"),
            )
    except FileNotFoundError:
        pass
