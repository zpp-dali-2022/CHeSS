import subprocess
import time
import glob
import os
import pandas as pd
import sys

# set working directory to the directory of a script
if os.path.dirname(sys.argv[0]):
    os.chdir(os.path.dirname(sys.argv[0]))


def get_data_path():
    try:
        return os.environ["DATA_PATH"]
    except KeyError:
        raise RuntimeError(
            "Missing data path, Please set the DATA_PATH environment variable"
        )


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


def save_results(results: pd.DataFrame, filename: str = "benchmark_results.csv"):
    header = False if os.path.exists(filename) else True
    results.to_csv(filename, mode="a", header=header)


def run_benchmark(benchmark):
    epochs = 5
    t = benchmark()
    results = pd.DataFrame(
        [
            {
                "benchmark": benchmark.__name__,
                "time": t,
                "throughput": n_images * epochs / t,
                "n_images": n_images,
            }
        ]
    )
    print(results)
    save_results(results)


if __name__ == "__main__":
    # path to the dataset to train on
    data_path = get_data_path()
    iterations = 10

    n_images = len(
        sorted(glob.glob(os.path.join(data_path, "*/fits/*.fits"), recursive=True))
    )

    for i in range(iterations):
        print(
            "========================================================================="
        )
        print(f"""\n\nIteration {i}\n\n""")
        print(
            "========================================================================="
        )

        run_benchmark(fitsio_benchmark)
        run_benchmark(dali_gpu_benchmark)
