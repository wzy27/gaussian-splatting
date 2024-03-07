import os
import subprocess, sys
import torchvision.transforms.functional as tf
from PIL import Image
from utils.image_utils import psnr
from tqdm import tqdm
import json
from pathlib import Path

# exp_dirs = [
#     "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/Omnizl",
# ]

# data_names = []

# for exp_dir in exp_dirs:
#     for data_name in os.listdir(exp_dir):
#         if os.path.exists(os.path.join(exp_dir, data_name, "meshes/00100000.ply")):
#             data_names.append(data_name)

# data_names.sort()
# print(data_names)
# print(len(data_names))

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTHONPATH"] = "$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu"

shell_command = "python chamferDistance.py -s data/CD_test/{0}/{0}-ours.ply -t data/CD_test/{0}/Scan-matched.obj"


with open("CD_done.json", "r") as done_file:
    done_dict = json.load(done_file)

for data_name in sorted(os.listdir("data/CD_test")):
    if data_name in done_dict:
        print(data_name, "done.")
    else:
        command = shell_command.format(data_name)
        print(command)
        subprocess.run(command, shell=True, executable="/bin/bash")
