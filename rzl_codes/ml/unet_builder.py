import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate, ReLU, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose


##  2D U-Net ###################################################################
################################################################################
def conv_block_2d(input_layer, num_filters, kernel_size=3):
    """
    Convolution block for 2D U-Net composed of two convolution layers with
    batch normalization and relu activation each time.

    input_layer: input
    num_filters: number of filters
    kernel_size: kernel size, defaulted to 3
    """
    x = Conv2D(num_filters, kernel_size, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(num_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def encoder_block_2d(input_layer, num_filters, conv_kernel_size=3,
                     pooling_size=2, dropout_rate=0):
    """
    Encoder block for 2D U-Net composed of convolution block, dropout layer,
    and MaxPooling layer.

    input_layer: input
    num_filters: number of filters in convolutional layers
    conv_kernel_size: kernel size of convolutional layers, defaulted to 3
    pooling_size: size of MaxPooling layer kernel, defaulted to (2,2)
    dropout_rate: dropout rate between convolution block and MaxPooling layer
    """
    x = conv_block_2d(input_layer, num_filters, kernel_size=conv_kernel_size)
    x = Dropout(dropout_rate)(x)
    p = MaxPooling2D(pool_size=pooling_size)(x)

    return x, p


def decoder_block_2d(input_layer, skipped_layer, num_filters,
                     conv_kernel_size=3, upsampling_type='upsampling',
                     pooling_size=2, dropout_rate=0):
    """
    Decoder block for 2D U-Net composed of UpSampling or Conv2DTranspose layer,
    concatenation with the less convoluted output of similar dimension, a
    convolution block, and a dropout layer

    input_layer: input
    skipped_layer: layer from encoding cascade with dimensions of current block
    num_filters: number of filters in convolutional layers
    conv_kernel_size: kernel size of convolutional layers, defaulted to 3
    upsampling_type: 'upsampling' or 'transpose'
    pooling_size: size of MaxPooling layer kernel, defaulted to (2,2)
    dropout_rate: dropout rate between convolution block and MaxPooling layer
    """

    if upsampling_type == "upsampling":
        x = UpSampling2D(size=pooling_size)(input_layer)
        x = Conv2D(num_filters, conv_kernel_size, padding="same")(x)
    elif upsampling_type == "transpose":
        x = Conv2DTranspose(num_filters, strides=pooling_size,
                            padding="same")(input_layer)
    x = Concatenate()([x, skipped_layer])
    x = conv_block_2d(x, num_filters, kernel_size=conv_kernel_size)
    x = Dropout(dropout_rate)(x)

    return x


def build_unet_2d(input_layer, encoding_layer_filters, kernel_size=3, num_classes=None, output_activation='sigmoid',
                  enc_block_dropout=0, bridge_dropout=0, dec_block_dropout=0):
    """
    Outputs the final layer of a 2D UNet.
    """

    skip_layers = []
    encoder_layers = []
    decoding_layers = []

    # Encoding
    e = input_layer  # Not actually encoded, just to initialize variable for loop
    for ilayer, num_filters in enumerate(encoding_layer_filters[:-1]):
        s, e = encoder_block_2d(e,
                                num_filters,
                                conv_kernel_size=kernel_size,
                                dropout_rate=enc_block_dropout)
        skip_layers.append(s)
        encoder_layers.append(e)

    # Bridge
    b1 = conv_block_2d(encoder_layers[-1],
                       encoding_layer_filters[-1],
                       kernel_size=kernel_size)
    b1 = Dropout(bridge_dropout)(b1)
    d1 = decoder_block_2d(b1,
                          skip_layers[-1],
                          encoding_layer_filters[-2],
                          conv_kernel_size=kernel_size,
                          dropout_rate=dec_block_dropout)
    decoding_layers.append(d1)

    # # Decoding
    for ilayer, num_filters in reversed(list(enumerate(encoding_layer_filters[:-2]))):
        d = decoder_block_2d(decoding_layers[-1],
                             skip_layers[ilayer],
                             num_filters,
                             conv_kernel_size=kernel_size,
                             dropout_rate=dec_block_dropout)
        decoding_layers.append(d)

    # # Activation
    if num_classes is None:
        output = Conv2D(input_layer.shape[-1], 1, padding="same", activation=output_activation)(decoding_layers[-1])
    else:
        output = Conv2D(num_classes, 1, padding="same", activation='sigmoid')(decoding_layers[-1])

    # Define model
    # model = Model(input_layer, output)

    return output

################################################################################
##  3D U-Net ###################################################################
################################################################################
def conv_block_3d(input_layer, num_filters, kernel_size=3):
    """
    Convolution block for 2D U-Net composed of two convolution layers with
    batch normalization and relu activation each time.

    input_layer: input
    num_filters: number of filters
    kernel_size: kernel size, defaulted to 3
    """
    x = Conv3D(num_filters, kernel_size, padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(num_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def encoder_block_3d(input_layer, num_filters, conv_kernel_size=3,
                     pooling_size=2, dropout_rate=0):
    """
    Encoder block for 2D U-Net composed of convolution block, dropout layer,
    and MaxPooling layer.

    input_layer: input
    num_filters: number of filters in convolutional layers
    conv_kernel_size: kernel size of convolutional layers, defaulted to 3
    pooling_size: size of MaxPooling layer kernel, defaulted to 2
    dropout_rate: dropout rate between convolution block and MaxPooling layer
    """
    x = conv_block_3d(input_layer, num_filters, kernel_size=conv_kernel_size)
    x = Dropout(dropout_rate)(x)
    p = MaxPooling3D(pool_size=pooling_size)(x)

    return x, p


def decoder_block_3d(input_layer, skipped_layer, num_filters,
                     conv_kernel_size=3, upsampling_type='upsampling',
                     pooling_size=2, dropout_rate=0):
    """
    Decoder block for 2D U-Net composed of UpSampling or Conv2DTranspose layer,
    concatenation with the less convoluted output of similar dimension, a
    convolution block, and a dropout layer

    input_layer: input
    skipped_layer: layer from encoding cascade with dimensions of current block
    num_filters: number of filters in convolutional layers
    conv_kernel_size: kernel size of convolutional layers, defaulted to 3
    upsampling_type: 'upsampling' or 'transpose'
    pooling_size: size of MaxPooling layer kernel, defaulted to (2,2)
    dropout_rate: dropout rate between convolution block and MaxPooling layer
    """

    if upsampling_type == "upsampling":
        x = UpSampling3D(size=pooling_size)(input_layer)
        x = Conv3D(num_filters, conv_kernel_size, padding="same")(x)
    elif upsampling_type == "transpose":
        x = Conv3DTranspose(num_filters, strides=pooling_size,
                            padding="same")(input_layer)
    x = Concatenate()([x, skipped_layer])
    x = conv_block_3d(x, num_filters, kernel_size=conv_kernel_size)
    x = Dropout(dropout_rate)(x)

    return x


def build_unet_3d(input_layer, encoding_layer_filters, kernel_size=3, 
                  enc_block_dropout=0, bridge_dropout=0, dec_block_dropout=0):

    skip_layers = []
    encoder_layers = []
    decoding_layers = []

    # Encoding
    e = input_layer  # Not actually encoded, just to initialize variable for loop
    for ilayer, num_filters in enumerate(encoding_layer_filters[:-1]):
        s, e = encoder_block_3d(e,
                                num_filters,
                                conv_kernel_size=kernel_size,
                                dropout_rate=enc_block_dropout)
        skip_layers.append(s)
        encoder_layers.append(e)

    # Bridge
    b1 = conv_block_3d(encoder_layers[-1],
                       encoding_layer_filters[-1],
                       kernel_size=kernel_size)
    b1 = Dropout(bridge_dropout)(b1)
    d1 = decoder_block_3d(b1,
                          skip_layers[-1],
                          encoding_layer_filters[-2],
                          conv_kernel_size=kernel_size,
                          dropout_rate=dec_block_dropout)
    decoding_layers.append(d1)

    # Decoding
    for ilayer, num_filters in reversed(list(enumerate(encoding_layer_filters[:-2]))):
        d = decoder_block_3d(decoding_layers[-1],
                             skip_layers[ilayer],
                             num_filters,
                             conv_kernel_size=kernel_size,
                             dropout_rate=dec_block_dropout)
        decoding_layers.append(d)

    # # Activation
    # if num_classes is None:
    #     output = Conv3D(input_shape[-1], 1, padding="same", activation=output_activation)(decoding_layers[-1])
    # else:
    #     output = Conv3D(num_classes, 1, padding="same", activation='sigmoid')(decoding_layers[-1])
    # 
    # # Define model
    # model = Model(input_layer, output)

    return decoding_layers[-1]
