import subprocess
import time
import glob
import os
import pandas as pd
import sys

# set working directory to the directory of a script
os.chdir(os.path.dirname(sys.argv[0]))

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


def save_results(results : pd.DataFrame, filename : str = "benchmark_results.csv"):
    header = False if os.path.exists(filename) else True
    df.to_csv(filename, mode="a", header=header)

if __name__ == "__main__":
    # path to the dataset to train on
    data_path = os.getenv("DATA_PATH")
    iterations = 10

    n_images = len(
        sorted(glob.glob(os.path.join(data_path, "*/fits/*.fits"), recursive=True))
    )
    results = {}

    for i in range(iterations):
        print("=========================================================================")
        print(f"""\n\nIteration {i}\n\n""")
        print("=========================================================================")
        for benchmark in [dali_gpu_benchmark, fitsio_benchmark]:
            t = benchmark()
            results[benchmark.__name__] = {
                "time": t,
                "throughput": n_images / t,
                "n_images": n_images,
            }

    df = pd.DataFrame(results)
    print(df)
    save_results(df)