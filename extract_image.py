import os, subprocess

name = "suitcase008"
dataset = "Omnizl"
img = 6

path_dicts = {
    "gt": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-base-SH/{dataset}/{name}/test/ours_30000/gt/r_{img}.png",
    "ours": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-ours-SH/{dataset}/{name}/test/ours_30000/renders/r_{img}.png",
    # "ours": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-0.010-200/{dataset}/{name}/test/ours_30000/renders/r_{img}.png",
    "3dgs": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-base-SH/{dataset}/{name}/test/ours_30000/renders/r_{img}.png",
    "gof": f"/data/nglm005/zhengyu.wen/gaussian-opacity-fields/output/{dataset}/{name}/test/ours_30000/test_preds_-1/{img:05}.png",
    "2dgs": f"/data/nglm005/zhengyu.wen/2d-gaussian-splatting/output-white/{dataset}/{name}/test/ours_30000/renders/{img:05}.png",
    "voxurf": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/render_omni/{name}/render_00{img}.png",
    "neus2": f"guest@10.88.72.201:/home/guest/data/zhengyu.wen/NeuS2/output/final-30000/{name}/test/render/ref_{img}.png",
}

methods = ["3dgs", "gt", "gof", "2dgs", "ours", "voxurf"]
save_root_path = "data/image_renders"
save_img_path = f"{save_root_path}/{name}-r_{img}"
os.makedirs(save_img_path, exist_ok=True)
for method in methods:
    source_path = path_dicts[method]
    target_path = f"{save_img_path}/{method}.png"
    command = f"cp {source_path} {target_path}"
    print(command)
    subprocess.run(command, shell=True, executable="/bin/bash")

# method = "neus2"
# command = f"scp {path_dicts[method]} {save_img_path}/{method}.png"
# print(command)
# subprocess.run(command, shell=True, executable="/bin/bash")
