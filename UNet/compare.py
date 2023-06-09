import subprocess
import time
import glob
import os
import pandas as pd


def dali_gpu_benchmark():
    start = time.time()
    subprocess.run(["python", "train.py", "--use_dali", "--GPU"], check=True)
    end = time.time()
    return end - start


def fitsio_benchmark():
    start = time.time()
    subprocess.run(["python", "train.py"], check=True)
    end = time.time()
    return end - start


if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")
    n_images = len(
        sorted(glob.glob(os.path.join(data_path, "*/fits/*.fits"), recursive=True))
    )
    results = {}

    for benchmark in [dali_gpu_benchmark, fitsio_benchmark]:
        time = benchmark()
        results[benchmark.__name__] = {
            "time": time,
            "throughput": n_images / time,
            "n_images": n_images,
        }

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv")
    print(df)
