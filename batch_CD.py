import os
import subprocess, sys
import torchvision.transforms.functional as tf
from PIL import Image
from utils.image_utils import psnr
from tqdm import tqdm
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTHONPATH"] = "$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu"

# shell_command = "python chamferDistance.py -s data/CD_test/{0}/{0}-ours.ply -t data/CD_test/{0}/Scan-matched.obj"
shell_command = "python chamferDistance.py -s data/CD_test/{0}/{0}-{1}.ply -t data/CD_test/{0}/Scan-matched.obj -d {1}"
# shell_command = "python chamferDistance.py -s data/CD_test/{0}/{0}-{1}.ply -t data/CD_test/{0}/gt_{0}.ply -d {1}"
division = "ours"

with open("CD_{}.json".format(division), "r") as done_file:
    done_dict = json.load(done_file)

for data_name in sorted(os.listdir("data/CD_test")):
    if data_name in done_dict:
        print(data_name, "done.")
    else:
        if os.path.exists("data/CD_test/{0}/Scan-matched.ply".format(data_name)):
            shell_command = "python chamferDistance.py -s data/CD_test/{0}/{0}-{1}.ply -t data/CD_test/{0}/Scan-matched.ply -d {1}"

        if os.path.exists("data/CD_test/{0}/{0}-{1}.ply".format(data_name, division)):
            # if os.path.exists("data/CD_test/{0}/gt_{0}.ply".format(data_name, division)):
            command = shell_command.format(data_name, division)
            print(command)
            subprocess.run(command, shell=True, executable="/bin/bash")
