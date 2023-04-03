import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import fitsio
from functools import partial

def read_image(path, output_size=(256, 256)):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, output_size)
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_fits_image(path, output_size=(256, 256)):
    x = fitsio.read(path)
    if output_size != x.shape:
        x = cv2.resize(x, output_size)
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path, output_size=(256, 256), normalize=False):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if output_size != x.shape:
        x = cv2.resize(x, output_size)
    # Masks are PNG files, data range already within [0-1]
    # Thus no need to normalize
    if normalize:
        x = x/255.0
    x = x.astype(np.float32)
    # Images as provided 3-channel RGB. Need to expand the masks' dimensions
    # to be consistent with the 3 dimensions in the images
    x = np.expand_dims(x, axis=-1)
    return x


def get_dataset_paths(dataset_path):
    images = sorted(glob(os.path.join(dataset_path, "images/*")))
    masks = sorted(glob(os.path.join(dataset_path, "masks/*")))

    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)

    return (train_x, train_y), (test_x, test_y)


def preprocess(image_path, mask_path, output_size=(256, 256)):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        x = read_image(image_path, output_size=output_size)
        y = read_mask(mask_path, output_size=output_size)

        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([*output_size, 3])
    mask.set_shape([*output_size, 1])

    return image, mask


def create_dataset(images, masks, batch=8, buffer_size=1000):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    # mapping when reading/processing images from paths should occur before batch()
    dataset = dataset.map(partial(preprocess, output_size=(256, 256)))
    # Batching with clear epoch separation (place repeat() after batch())
    # https://www.tensorflow.org/guide/data#training_workflows
    dataset = dataset.batch(batch).repeat()
    # Prefetch timing effetcs: prefetch(2) => 622s with 512x512x3 & batch size 16. 
    # dataset = dataset.prefetch(2)
    return dataset
