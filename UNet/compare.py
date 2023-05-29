import subprocess
import time

print()
print("=========================================================================")
print("================================ DALI, CPU ==============================")
print("=========================================================================")
print()
start = time.time()
subprocess.run(["python", "train.py", "--use_dali"], check=True)
end = time.time()
dali_cpu_time = end - start

print()
print("=========================================================================")
print("================================ DALI, GPU ==============================")
print("=========================================================================")
print()
start = time.time()
subprocess.run(["python", "train.py", "--use_dali", "--GPU"], check=True)
end = time.time()
dali_gpu_time = end - start

print()
print("=========================================================================")
print("============================= FITSIO, GPU ===========================")
print("=========================================================================")
print()
start = time.time()
subprocess.run(["python", "train.py"], check=True)
end = time.time()
FITSIO_gpu_time = end - start

print()
print("=========================================================================")
print("======================= RESULTS FOR BATCH SIZE: 4 =======================")
print("=========================================================================")
print()

print(f"Time for DALI (CPU): {dali_cpu_time} s")
print(f"Time for DALI (GPU): {dali_gpu_time} s")
print(f"Time for FITSIO (GPU): {FITSIO_gpu_time} s")

# how many seconds faster
diff_cpu_gpu = dali_cpu_time - dali_gpu_time
print(f"DALI on GPU is {diff_cpu_gpu} s faster than DALI on CPU")
diff_s =  FITSIO_gpu_time - dali_gpu_time
print(f"DALI on GPU is {diff_s} s faster than FITSIO on GPU")

# how many times faster:
print(f"DALI on GPU is {dali_cpu_time / dali_gpu_time} times faster than DALI on CPU")
print(f"DALI on GPU is {FITSIO_gpu_time / dali_gpu_time} times faster than FITSIO on GPU")
print("=========================================================================")

with open('auto_compare_results.txt', 'w') as file:
    # print to file everything that was printed to console
    print("=========================================================================", file=file)
    print("RESULTS FOR BATCH SIZE: 4", file=file)

    print(f"Time for DALI (CPU): {dali_cpu_time} s", file=file)
    print(f"Time for DALI (GPU): {dali_gpu_time} s", file=file)
    print(f"Time for FITSIO (GPU): {FITSIO_gpu_time} s", file=file)

    # how many seconds faster
    print(f"DALI on GPU is {diff_cpu_gpu} s faster than DALI on CPU", file=file)
    print(f"DALI on GPU is {diff_s} s faster than FITSIO on GPU", file=file)

    # how many times faster:
    print(f"DALI on GPU is {dali_cpu_time / dali_gpu_time} times faster than DALI on CPU", file=file)
    print(f"DALI on GPU is {FITSIO_gpu_time / dali_gpu_time} times faster than FITSIO on GPU", file=file)
    print("=========================================================================", file=file)
