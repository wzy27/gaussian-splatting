import os
import subprocess, sys

exp_dirs = [
    # "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/Omnizl",
    "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/blender",
]

data_dirs = []

for exp_dir in exp_dirs:
    for data_name in os.listdir(exp_dir):
        # if not os.path.exists(os.path.join(exp_dir, data_name, "result-octree.txt")):
        data_dirs.append(os.path.join(exp_dir, data_name))

data_dirs.sort()
print(data_dirs)
print(len(data_dirs))

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTHONPATH"] = "$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu"
print(os.environ["PYTHONPATH"])

shell_command_prefix = "python train-octree.py --eval -s "
shell_command_postfix = " --sh_degree 0 --iterations 15000 --save_iterations 15000 "
shell_param = "--lambda_opacity 3 --lambda_orientation 0.1 --lambda_scale 1 --opacity_reset_interval 300 -w"

for data_dir in data_dirs:
    command = shell_command_prefix + data_dir + shell_command_postfix + shell_param
    print(command)
    subprocess.run(command, shell=True, executable="/bin/bash")
    with open("batch_octree.log", "a+") as log_file:
        log_file.write(data_dir + " Done.\n")
