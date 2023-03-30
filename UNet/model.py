from tensorflow.keras import layers, Model


def conv_block(inputs, n_filters, ksize=3, kinit='HeNormal', activation=True, batch_norm=True):
    """
    Block of convolution with optional BN and activation
    see unet_model() for description of full input list
    """
    x = layers.Conv2D(n_filters, ksize, padding='same', kernel_initializer=kinit)(inputs)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    return x


def contracting_miniblock(inputs, n_filters, ksize=3, kinit='HeNormal',
                          batch_norm=True, max_pooling=True, res_block=False, dropout=0):
    """
    Elementary block of the encoder. Can set the number of convolution blocks before max pooling with `n_layers`.
    Can toggle BN, max pooling, activation, res block.
    see unet_model() for description of full input list
    """
    x = inputs
    x = conv_block(x, n_filters, ksize=ksize, kinit=kinit, batch_norm=batch_norm, activation=True)

    if res_block:
        x = conv_block(x, n_filters, ksize=ksize, kinit=kinit, batch_norm=batch_norm, activation=False)
        res = layers.Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same', kernel_initializer=kinit)(inputs)
        x = layers.Add()([x, res])
        x = layers.Activation('relu')(x)
    else:
        x = conv_block(x, n_filters, ksize=ksize, kinit=kinit, batch_norm=batch_norm, activation=True)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    # Use the unpooled layer output for skip connection, make it the 2nd output.
    skip_connection = x
    if max_pooling:
        output = layers.MaxPooling2D(pool_size=(2, 2))(x)
    else:
        output = x

    return output, skip_connection


def expanding_miniblock(inputs, skip_connection,
                        n_filters=64, ksize=3, kinit='HeNormal',
                        activation=True, transpose_conv=False, batch_norm=True, res_block=False):
    """
    Elementary block of the decoder. With optional activation, transpose convolution instead of UpSampling + conv, BN.
    see unet_model() for description of full input list
    """

    # Start with upsampling to double the size of the image
    # Upsampling kernel_size fixed at 2, different from the user-set kernel sizes, due to size requirements
    if transpose_conv:
        up = layers.Conv2DTranspose(n_filters, kernel_size=2, strides=(2, 2), padding='same')(inputs)
    else:
        up = layers.UpSampling2D((2, 2), interpolation="bilinear")(inputs)
        up = layers.Conv2D(n_filters, kernel_size=2, padding='same', kernel_initializer=kinit)(up)
    # Merge the skip connection from previous block to prevent information loss (cropping if needed)
    merge = layers.concatenate([up, skip_connection], axis=3)
    # Add 2 Conv layers with relu activation and HeNormal initialization
    conv = conv_block(merge, n_filters, ksize=ksize, kinit=kinit, activation=activation, batch_norm=batch_norm)
    if res_block:
        conv = conv_block(conv, n_filters, ksize=ksize, kinit=kinit, batch_norm=batch_norm, activation=False)
        res = layers.Conv2D(n_filters, (1, 1), strides=(1, 1), padding='same', kernel_initializer=kinit)(merge)
        conv = layers.Add()([res, conv])
        conv = layers.Activation('relu')(conv)
    else:
        conv = conv_block(conv, n_filters, ksize=ksize, kinit=kinit, batch_norm=batch_norm, activation=True)

    return conv


def unet_model(input_shape, n_classes, filters, compiler_dict,
               ksize=3, batch_norm=True, max_pools=(True, True, True, True, False),
               kinit='HeNormal', res_block=False, transpose_conv=False, final_activation='sigmoid',
               dropout=(0, 0, 0, 0, 0)):

    """
    Creates the U-Net model made of:
    (1) contracting branch: stack of contracting, downsampling mini blocks
    (2) expanding branch: stack of expanding, upsampling mini blocks
    (3) A convolution that brings back the number of filters to the number of classes

    :param input_shape: tuple of dimensions for the input batch of images e.g. (32, 256, 256, 3)
    :param n_classes: number of classes. Set it to 1 for binary classification.
    :param filters: Tuple of number of filters, the last element represents the brige of the U-Net.
    :param compiler_dict: dictionary of parameters for the model compiler
    :param ksize: kernel sizes for the convolution
    :param batch_norm: toggle batch normalization
    :param max_pools: tuple of True/False to enable/disable max pooling in encoder block. You must have as many
                        elements as in the layer_filters.
    :param kinit: kernel initializer
    :param res_block: toggle the use of a ResNet block
    :param transpose_conv: determines whether you upsample with transpose convolution or with UpSampling2D + convolution
    :param final_activation: output layer activation function
    """

    inputs = layers.Input(shape=input_shape)

    # Contracting branch: stack the encoder mini blocks
    skips = []
    x = [inputs, 0]
    for nf, mp, do in zip(filters, max_pools, dropout):
        x = contracting_miniblock(x[0], n_filters=nf, ksize=ksize, kinit=kinit,
                                  batch_norm=batch_norm, max_pooling=mp, res_block=res_block,
                                  dropout=do)
        skips.append(x[1])

    # Exanding branch: decoder stacks the upsampling mini blocks with decreasing nb of filters.
    x = skips[-1]
    skips = reversed(skips[:-1])
    # Nb of filters for the decoder branch. Derived from those in the encoder branch
    up_filters = reversed(filters[:-1])
    for i, (nf, skip) in enumerate(zip(up_filters, skips)):
        x = expanding_miniblock(x, skip, n_filters=nf, ksize=ksize, kinit=kinit, batch_norm=batch_norm,
                                transpose_conv=transpose_conv, res_block=res_block)

    # This layer is present in the original implementation. On RGB people images, turned out to give much
    # less accuracy. Removing for now, but may need some testing on the medical imagery samples, closer to sun-like images
    # if not res_block:
    #     x = layers.Conv2D(2, 3, activation='relu', padding='same')(x)
    # complete the model with a 1x1 conv layer with the number of filters equal to the number of classes
    x = layers.Conv2D(n_classes, 1, activation=final_activation, padding='same')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x)

    # compiler
    model.compile(**compiler_dict)

    return model
