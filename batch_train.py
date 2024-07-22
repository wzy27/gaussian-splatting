import os
import subprocess, sys

exp_dirs = [
    "Omnizl",
    "blender",
]

data_dirs = []

for exp_dir in exp_dirs:
    for data_name in os.listdir(os.path.join("data", exp_dir)):
        # if not os.path.exists(os.path.join(exp_dir, data_name, "result-base.txt")):
        data_dirs.append((exp_dir, data_name))
        # else:
        #     print(data_name)

print(data_dirs)
data_dirs.sort()

os.environ["PYTHONPATH"] = "$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu"
print(os.environ["PYTHONPATH"])

for exp_dir, data_dir in data_dirs:
    if not os.path.exists(
        os.path.join("output-base-SH", exp_dir, data_dir, "base-result.json")
    ):
        command = "./quick_train.sh {} {} {}".format(exp_dir, data_dir, 4)
        print(command)
        subprocess.run(command, shell=True, executable="/bin/bash")
        # with open("batch_octree.log", "a+") as log_file:
        #     log_file.write(data_dir + " Done.\n")
