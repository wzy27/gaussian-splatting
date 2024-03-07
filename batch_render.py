import os
import subprocess, sys

exp_dirs = [
    "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/Omnizl",
    "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/blender",
]

data_dirs = []

for exp_dir in exp_dirs:
    for data_name in os.listdir(exp_dir):
        result_base_path = os.path.join(exp_dir, data_name, "result-base.txt")
        result_octree_path = os.path.join(exp_dir, data_name, "result-octree.txt")
        if os.path.exists(result_base_path) and os.path.exists(result_octree_path):
            with open(result_base_path, "r") as f1, open(result_octree_path, "r") as f2:
                result_base = f1.read().splitlines()
                psnr_base = float(result_base[1].split(" ")[1])
                points_base = int(result_base[2].split(" ")[1])

                result_octree = f2.read().splitlines()
                psnr_octree = float(result_octree[1].split(" ")[1])
                points_octree = int(result_octree[2].split(" ")[1])
                # print(psnr)

                if (
                    psnr_octree - psnr_base > -1
                    and (points_octree - points_base) / points_base < -0.2
                    and not os.path.exists(
                        os.path.join(exp_dir, data_name, "render_base_test_black")
                    )
                ):
                    data_dirs.append(os.path.join(exp_dir, data_name))


print(data_dirs)
data_dirs.sort()

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

shell_command_prefix = "python render.py -s "
shell_command_postfix = " -m /data/nglm005/zhengyu.wen/final/gaussian-splatting/output/03-05-22:20:11 --iteration 15000 --skip_train"
# shell_param = "--lambda_opacity 10 --lambda_orientation 1 --lambda_scale 1000"

for data_dir in data_dirs:
    command = shell_command_prefix + data_dir + shell_command_postfix
    print(command)
    subprocess.run(command, shell=True, executable="/bin/bash")
    # with open("batch_train.log", "a+") as log_file:
    #     log_file.write(data_dir + " Done.\n")
