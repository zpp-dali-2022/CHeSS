import os
import numpy as np

# Convert '.npz' files to '.npy' format
def convert_npz_to_npy(path):
    npz_files = [file for file in os.listdir(path) if file.endswith('.npz')]
    for npz_file in npz_files:
        npz_file_path = os.path.join(path, npz_file)
        npy_file_path = os.path.join(path, npz_file.replace('.npz', '.npy'))
        data = np.load(npz_file_path)
        np.save(npy_file_path, data)
    return

data_path = os.getenv('DATA_PATH', '/home/aderylo/2011/01/')

convert_npz_to_npy(os.path.join( data_path, "label"))