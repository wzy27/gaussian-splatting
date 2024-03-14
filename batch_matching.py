import os
import subprocess, sys
from pathlib import Path

base_dir = "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data"

exp_dirs = [
    "Omnizl",
    # "blender",
]

data_dirs = []

for exp_dir in exp_dirs:
    for data_name in os.listdir(os.path.join(base_dir, exp_dir)):
        ours_path = os.path.join(base_dir, exp_dir, data_name, "meshes/00100000.ply")

        done_path = os.path.join("data/CD_test", data_name, "Scan-matched.ply")
        done_path_2 = os.path.join("data/CD_test", data_name, "Scan-matched.obj")
        # if os.path.exists(ours_path) and not (
        #     os.path.exists(done_path) or os.path.exists(done_path_2)
        # ):
        #     data_dirs.append(os.path.join(exp_dir, data_name))
        if os.path.exists(ours_path):
            data_dirs.append(os.path.join(exp_dir, data_name))
        # else:
        #     print(data_name)

data_dirs.sort()
print(data_dirs)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

shell_command = "python mesh_matching.py -s {} -t {} -d {}"
# shell_param = "--lambda_opacity 10 --lambda_orientation 1 --lambda_scale 1000"

fail_list = []

for data_dir in data_dirs:
    data_name = Path(data_dir).stem

    load_path = {
        "ours": os.path.join(base_dir, data_dir, "meshes/00100000.ply"),
        "gt": os.path.join(base_dir, "gt", data_name, "Scan/Scan.obj"),
        "voxurf": os.path.join(base_dir, "voxurf", "{}.ply".format(data_name)),
    }
    save_path = {
        "ours": os.path.join(
            base_dir, "CD_test", data_name, "{}-ours.ply".format(data_name)
        ),
        "gt": os.path.join(base_dir, "CD_test", data_name, "Scan-matched.ply"),
        "voxurf": os.path.join(
            base_dir, "CD_test", data_name, "{}-voxurf.ply".format(data_name)
        ),
    }
    degree = {
        "ours": 270,
        "gt": 0,
        "voxurf": 270,
    }

    gt_path_2 = os.path.join("data/CD_test", data_name, "Scan-matched.obj")

    done_path = os.path.join("data/CD_test", data_name)
    os.makedirs(done_path, exist_ok=True)
    for division in load_path.keys():
        command = shell_command.format(
            load_path[division], save_path[division], degree[division]
        )
        if not os.path.exists(save_path[division]):
            if division != "gt" or not os.path.exists(gt_path_2):
                fail_list.append(os.path.join(data_name, division))
                print(command)
                subprocess.run(command, shell=True, executable="/bin/bash")

print(fail_list)
# ours_path = os.path.join(base_dir, data_dir, "meshes/00100000.ply")
# ours_save_path = os.path.join(
#     base_dir, "CD_test", data_name, "{}-ours.ply".format(data_name)
# )
# gt_path = os.path.join(base_dir, "gt", data_name, "Scan/Scan.obj")
# gt_save_path = os.path.join(base_dir, "CD_test", data_name, "Scan-matched.ply")
# done_path = os.path.join("data/CD_test", data_name)
# os.makedirs(done_path, exist_ok=True)

# command_ours = shell_command.format(ours_path, ours_save_path, 270)
# command_gt = shell_command.format(gt_path, gt_save_path, 0)

# if not os.path.exists(ours_save_path):
#     print(command_ours)
#     subprocess.run(command_ours, shell=True, executable="/bin/bash")

# if not os.path.exists(gt_save_path):
#     print(command_gt)
#     subprocess.run(command_gt, shell=True, executable="/bin/bash")
