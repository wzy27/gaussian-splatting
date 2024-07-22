# training scripts for the nerf-synthetic datasets
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools

dataset_dirs = [
    # "mechparts-renders-smooth-ENV",
    # "mechparts-renders-ENV",
    # "mechparts-renders-play-ENV",
    "oo3d-render",
]

data_dirs = []
for exp_dir in dataset_dirs:
    for data_name in os.listdir(os.path.join("data", exp_dir)):
        # if not os.path.exists(os.path.join(exp_dir, data_name, "result-octree.txt")):
        data_dirs.append((exp_dir, data_name))

data_dirs.sort()
data_dirs.reverse()
print(data_dirs)
print(len(data_dirs))

output_dir = "output-cad"

excluded_gpus = set([0, 1, 2, 3, 7])
dry_run = False
factors = [1]
jobs = list(itertools.product(data_dirs, factors))


def train_scene(gpu, data_dir, factor):
    dataset_dir, instance_name = data_dir
    if os.path.exists(
        os.path.join(output_dir, dataset_dir, instance_name, "base-result.json")
    ):
        cmd = f"./quick_render.sh {dataset_dir} {output_dir} {instance_name} {gpu}"
        print(cmd)
        if not dry_run:
            os.system(cmd)

    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5)
        )

        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        # print(available_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(
                worker, gpu, *job
            )  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(
                future
            )  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
