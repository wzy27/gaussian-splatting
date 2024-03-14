import os, shutil


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


source_dir = "data/Omnizl"
target_dir = "data/Omni-clean"

os.makedirs(target_dir, exist_ok=True)

splits = ["train", "test", "val"]
for data_name in sorted(os.listdir(source_dir)):
    os.makedirs(os.path.join(target_dir, data_name), exist_ok=True)
    for split in splits:
        json_source_path = os.path.join(
            source_dir, data_name, "transforms_{}.json".format(split)
        )
        json_target_path = os.path.join(
            target_dir, data_name, "transforms_{}.json".format(split)
        )
        image_source_dir = os.path.join(source_dir, data_name, split)
        image_target_dir = os.path.join(target_dir, data_name, split)

        # copytree(image_source_dir, image_target_dir)
        shutil.copytree(image_source_dir, image_target_dir)
        shutil.copy(json_source_path, json_target_path)
    print(data_name, "Done.")
