import os
import glob
import tensorflow as tf
import nvidia.dali.fn as fn
import argparse
import data
import model
import matplotlib.pyplot as plt

# Added '--use_dali' optional option and '--GPU' option (in case of using DALI)
parser = argparse.ArgumentParser()
parser.add_argument('--use_dali', action='store_true', help='Use DALI for processing')
parser.add_argument('--GPU', action='store_true', help='Use GPU for computing')
args = parser.parse_args()
# Restrict GPU memory to 45 GB:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024*45)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
else:
    print('Using CPUs')

# -------- Setup training/validation/test data -------------#
normalize_image = True  # Will normalize between [0-1]
normalize_masks = False  # Masks are already between [0-1]
input_shape = (256, 256, 1)  # image dimensions, nb of channels. Set 1 channel for SDO data if testing 1 wavelength
batch_size = 16  # Note that we will use an infinitely repeating data generator
if args.use_dali:
    # Paths to images and label masks
    data_path = os.getenv('DATA_PATH', '/home/ahess/2011/01/')
    images = sorted(glob.glob(os.path.join(data_path, "fits/*.fits" )))
    masks = sorted(glob.glob(os.path.join(data_path, "label/*.npy")))
    use_GPU = args.GPU

    # Generate training/test data
    train_dataset, test_dataset, n_train, n_test = data.create_train_test_sets(images, masks, input_shape,
                                                                            normalize_images=True,
                                                                            normalize_masks=False,
                                                                            batch_size=batch_size,
                                                                            buffer_size=5000, use_dali = True, use_GPU = use_GPU)          
                                                                            
else:    
    # Paths to images and label masks
    data_path = os.getenv('DATA_PATH', '/home/ahess/2011/01/')
    images = sorted(glob.glob(os.path.join(data_path, "fits/*.fits" )))
    masks = sorted(glob.glob(os.path.join(data_path, "label/*.npz")))


    # Generate training/test data
    train_dataset, test_dataset, n_train, n_test = data.create_train_test_sets(images, masks, input_shape,
                                                                            normalize_images=True,
                                                                            normalize_masks=False,
                                                                            batch_size=batch_size,
                                                                            buffer_size=5000, use_dali = False, use_GPU = False)
                                                                 


# -------------------------------------------------------------------------------- #
# ------------------------- Define the U-Net ------------------------------------- #
# -------------------------------------------------------------------------------- #
checkpoint_dir = 'checkpoints'
checkpoint_name = f'final_checkpoint.ckpt'  # file containing the trained weights after last epoch
# Use pretrained weights if needed. Set to None to train from scratch
pretrained_weights = None  # tf.train.latest_checkpoint(checkpoint_dir)

n_classes = 1   # Number of classes. If binary, set n_classes to 1, used as the number of filters for the last layer

# nb of filters in each layer
# The last encoder mini block is the bridge; it does not use max pooling
filters = [64, 128, 256, 512, 1024]
# Max pooling set as True/False, each element applies to each mini block of the encoder
# You must have as many element as the filters
max_pools = (True, True, True, True, False)

ksize = 3           # Kernel size for the convolution blocks
kinit = 'HeNormal'  # Kernel initializer
batch_norm = True   # Toggle Batch Normalization
res_block = True   # Toggle ResNet block
transpose_conv = False  # Set whether to use Transpose conv. instead of Upsampling + conv
# Activation function for the output layer: some use softmax. Tested on people segmentation showed poor accuracy
final_activation = 'sigmoid'
# -------------------------------------------------------------------------------- #
# -------------------------Training parameters ----------------------------------- #
# -------------------------------------------------------------------------------- #
# As we use a generator with random augmentation, train lengths and batch size are not constraining the
# number of steps per epochs and validation steps. Nonetheless, pick reasonable numbers to avoid overtraining
# given that there are only 30 samples in the data source that are very similar to each other.
epochs = 10
steps_per_epochs = n_train//batch_size
# validation steps to perform at the end of each epoch:
# with the dataset generator, must be set to avoid an infinite evaluation loop after each training epoch
validation_steps = n_test//batch_size

# Optimizer parameters #
learning_rate = 1e-3    # This is the default value for the 'Adam' optimizer
compiler_dict = dict(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Instantiate the model, fit and save weights#
unet = model.unet_model(input_shape, n_classes, filters, compiler_dict,
                        ksize=3,
                        batch_norm=batch_norm,
                        max_pools=max_pools,
                        kinit=kinit,
                        res_block=res_block,
                        final_activation=final_activation,
                        transpose_conv=transpose_conv)


if pretrained_weights is not None:
    unet.load_weights(pretrained_weights)

history = unet.fit(train_dataset,
                   epochs=epochs,
                   steps_per_epoch=steps_per_epochs,
                   validation_data=test_dataset,
                   validation_steps=validation_steps)

checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
unet.save_weights(checkpoint_path)

unet.save(f'trainedUNet_resNet_{int(res_block)}')

# plot training and validation accuracy
plt.figure(figsize=(8, 8))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.savefig(f'U-Net_accuracies_resNet_{int(res_block)}.png')
