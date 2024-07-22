import os, json
import subprocess, sys

with open("data/need_names.json", "r") as f:
    name_dict = json.load(f)

# methods = ["ours", "base"]
# for method in methods:
for name in name_dict:
    dataset = "blender"
    if name[-1].isdigit():
        dataset = "Omnizl"
        continue

    command = f"python render.py -s data/{dataset}/{name} -m output-0.010-200/{dataset}/{name} --iteration 30000 --skip_train -w"
    print(command)
    subprocess.run(command, shell=True, executable="/bin/bash")
    # with open("batch_train.log", "a+") as log_file:
    #     log_file.write(data_dir + " Done.\n")
