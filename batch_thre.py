import os
import subprocess, sys
import json

op_lists = [0.1, 0.3, 1]

output_dir = "output-hyperparam"
dataset_dir = "blender"

for op in op_lists:
    save_name = f"OP{op:.2f}-init"
    for instance in os.listdir(os.path.join("data", dataset_dir)):
        command = "./threshold_train.sh {} {} {} {} {}".format(
            dataset_dir, output_dir, instance, 3, op
        )
        print(command)
        subprocess.run(command, shell=True, executable="/bin/bash")
