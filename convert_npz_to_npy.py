import os
import numpy as np

# Convert '.npz' files to '.npy' format
def convert(path):
    npz_files = [file for file in os.listdir(path) if file.endswith('.npz')]
    for npz_file in npz_files:
        npz_file_path = os.path.join(path, npz_file)
        npy_file_path = os.path.join(path,
                                     npz_file.replace('.npz', '.npy'))

        with np.load(npz_file_path) as data:
            np.save(npy_file_path, data['arr_0'].astype(np.uint8))


if __name__ == "__main__":
    path = os.getenv('DATA_PATH', '/home/ahess/2011/01/')
    path = os.path.join( path, "label")
    convert(path)