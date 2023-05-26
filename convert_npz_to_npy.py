import os
import numpy as np

# Convert '.npz' files to '.npy' format
def convert(path):
    npz_files = [file for file in os.listdir(path) if file.endswith('.npz')]
    for npz_file in npz_files:
        npz_file_path = os.path.join(path, npz_file)
        npy_file_path = os.path.join(path,
                                     npz_file.replace('.npz', '.npy'))
        data = np.load(npz_file_path)
        np.save(npy_file_path, data['arr_0'])
    return