import os
import numpy as np

# Funkcja rekurencyjna do konwersji plików '.npz' na format '.npy'
def convert_recursive(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_file_path = os.path.join(root, file)
                npy_file_path = os.path.join(root, file.replace('.npz', '.npy'))

                with np.load(npz_file_path) as data:
                    np.save(npy_file_path, data['arr_0'].astype(np.uint8))


if __name__ == "__main__":
    directory = os.getenv('DATA_PATH', '/home/mpalkus/2011/')
    convert_recursive(directory)
