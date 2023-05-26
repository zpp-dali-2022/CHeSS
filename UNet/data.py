import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import fitsio
from functools import partial
from pathlib import Path
import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn



def preprocess_array(arr,  normalize, output_size=(256, 256)):
    if output_size != arr.shape:
        cv2.setNumThreads(1)
        arr = cv2.resize(arr, output_size)
    if normalize:
        arr_n = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr_n = arr
    arr_n = arr_n.astype(np.float32)
    return arr_n


def preprocess_image(path, normalize, output_size=(256, 256)):
    if Path(path).suffix == '.fits':
        x = fitsio.read(path)
    else:
        x = cv2.imread(path, cv2.IMREAD_COLOR)

    x = preprocess_array(x, normalize, output_size=output_size)
    # If the image has only 1 channel (not a 3D cube), we need to add another dimension
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=-1)
    return x


def preprocess_mask(path, normalize, output_size=(256, 256)):
    if Path(path).suffix == '.npz':
        data = np.load(path)
        x = data['arr_0'].astype(np.uint8)
    elif Path(path).suffix == '.npy':   
        data = np.load(path)
        x = data['arr_0'].astype(np.uint8) 
    else:
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = preprocess_array(x, normalize, output_size=output_size)
    # When images are provided with 3-channel RGB, need to expand the masks' dimensions
    # to be consistent with the 3 dimensions in the images
    x = np.expand_dims(x, axis=-1)
    return x


def split_dataset_paths(images, masks):
    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)
    return (train_x, train_y), (test_x, test_y)


def preprocess(image_path, mask_path, input_shape, normalize_images, normalize_masks):
    def f(im_path, m_path):
        im_path = im_path.decode()
        m_path = m_path.decode()

        x = preprocess_image(im_path, normalize_images, output_size=(input_shape[0], input_shape[1]))
        y = preprocess_mask(m_path, normalize_masks, output_size=(input_shape[0], input_shape[1]))

        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape(input_shape)
    mask.set_shape([input_shape[0], input_shape[1], 1])

    return image, mask


def create_dataset(images, masks, input_shape, normalize_images, normalize_masks, batch=8, buffer_size=1000):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    # mapping when reading/processing images from paths should occur before batch()
    dataset = dataset.map(partial(preprocess,
                                  input_shape=input_shape,
                                  normalize_images=normalize_images,
                                  normalize_masks=normalize_masks),
                          num_parallel_calls=tf.data.AUTOTUNE)
    # Batching with clear epoch separation (place repeat() after batch())
    # https://www.tensorflow.org/guide/data#training_workflows
    # The preprocessing includes the reading of the files, expensive for big files. Thus put cache() after that.
    dataset = dataset.cache().batch(batch).repeat()
    # Prefetch timing effetcs: prefetch(2) => .jpg: no difference with/without: 622s at 512x512x3, batch size 16
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

import nvidia.dali.plugin.tf as dali_tf

def create_dataset_DALI(images, masks, input_shape, normalize_images, normalize_masks, batch=8, buffer_size=1000):
    shapes = (
        (batch, input_shape[0], input_shape[1], 1),
        (batch, input_shape[0], input_shape[1], 1)
    )
    dtypes = (tf.float32, tf.float32)

    data_path = os.getenv('DATA_PATH', '/home/aderylo/2011/01/')
    
    with tf.device('/cpu:0'):
        created_dataset = dali_tf.DALIDataset(
            pipeline=pipe(images, masks, os.path.join(data_path), input_shape),
            batch_size=batch,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=0) 

    return created_dataset   

# DALI pipeline
import nvidia.dali.types as types

@pipeline_def(device_id=0, batch_size=8)
def pipe(images, masks, path, input_shape, device="cpu", file_list=None, files=None,
                       hdu_indices=None, dtype=float):
    images = fn.experimental.readers.fits(device=device, files=images,
                                          file_root=path, file_filter="*.npy", shard_id=0,
                                          num_shards=1)
    images = fn.cast(images, dtype=types.FLOAT)
    masks = fn.readers.numpy(device=device,
                             files=masks,
                             file_root=path,
                             file_filter="*.fits",
                             shard_id=0,
                             num_shards=1)
                      
    images = fn.resize(images, resize_x=256, resize_y=256)
    masks = fn.resize(masks, resize_x=256, resize_y=256)
    
    images = fn.normalize(images, dtype=types.FLOAT)
    
    return images, masks






def create_train_test_sets(images, masks, input_shape, normalize_images, normalize_masks,
                           batch_size=8, buffer_size=1000, use_dali=False):

    (train_x, train_y), (test_x, test_y) = split_dataset_paths(images, masks)

    if (use_dali == False):
        train_dataset = create_dataset(train_x, train_y, input_shape, normalize_images, normalize_masks,
                                    batch=batch_size,
                                    buffer_size=buffer_size)
        test_dataset = create_dataset(test_x, test_y, input_shape, normalize_images, normalize_masks,
                                    batch=batch_size,
                                    buffer_size=buffer_size)
    else:
        #use DALI instead of astropy
        train_dataset = create_dataset_DALI(train_x, train_y, input_shape, normalize_images, normalize_masks,
                            batch=batch_size,
                            buffer_size=buffer_size)
        test_dataset = create_dataset_DALI(test_x, test_y, input_shape, normalize_images, normalize_masks,
                                    batch=batch_size,
                                    buffer_size=buffer_size)

    n_train = len(train_x)
    n_test = len(test_x)

    return train_dataset, test_dataset, n_train, n_test

# Convert '.npz' files to '.npy' format
def convert_npz_to_npy(path):
    npz_files = [file for file in os.listdir(path) if file.endswith('.npz')]
    for npz_file in npz_files:
        npz_file_path = os.path.join(path, npz_file)
        npy_file_path = os.path.join(path, npz_file.replace('.npz', '.npy'))
        data = np.load(npz_file_path)
        np.save(npy_file_path, data)
    return    
