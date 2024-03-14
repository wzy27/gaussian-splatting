import os
import subprocess, sys

exp_dirs = [
    # "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/Omnizl",
    "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/blender",
]

data_dirs = []

for exp_dir in exp_dirs:
    for data_name in os.listdir(exp_dir):
        # if not os.path.exists(os.path.join(exp_dir, data_name, "result-base.txt")):
        data_dirs.append(os.path.join(exp_dir, data_name))
        # else:
        #     print(data_name)

print(data_dirs)
data_dirs.sort()

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

shell_command_prefix = "python train-base.py --eval -s "
shell_command_postfix = " --sh_degree 0 --iterations 30000 --save_iterations 30000 -w"
# shell_param = "--lambda_opacity 10 --lambda_orientation 1 --lambda_scale 1000"

for data_dir in data_dirs:
    command = shell_command_prefix + data_dir + shell_command_postfix
    print(command)
    subprocess.run(command, shell=True, executable="/bin/bash")
    with open("batch_train.log", "a+") as log_file:
        log_file.write(data_dir + " Done.\n")
