# utf-8

from keras.layers import Input, Dense, Dropout, Activation, Conv2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model

import keras.backend as K

K.set_image_data_format('channels_last')


def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    #
    concat_axis = -1
    eps = 0.001
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Conv2D(inter_channel, kernel_size=(1, 1), padding='same', use_bias=False)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = Conv2D(nb_filter, kernel_size=(3, 3), padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(rate=dropout_rate)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None):
    #
    concat_axis = -1
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis)

        nb_filter += growth_rate

    return concat_feat, nb_filter


def transition_block(x, stage, nb_filter, compression=1.0):
    #
    concat_axis = -1
    eps = 0.001
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Activation('relu', name=relu_name_base)(x)

    # compression
    nb_filter = int(nb_filter * compression)
    x = Conv2D(nb_filter, kernel_size=(1, 1), padding='same', use_bias=False)(x)

    # avg_pool
    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x, nb_filter


def densenet_builder(config, nb_layers, nb_dense_block=4, reduction=0.5):
    #
    concat_axis = -1
    eps = 0.001

    # compute compression factor
    compression = 1.0 - reduction

    input_shape = (config.img_size, config.img_size, config.channels)
    img_input = Input(shape=input_shape, name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    dropout_rate = config.dropout_rate
    nb_filter = 64

    # Initial convolution
    x = Conv2D(nb_filter, (7, 7), strides=2, padding='same', use_bias=False)(img_input)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Activation('relu', name='relu1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter,
                                   growth_rate=config.growth_rate, dropout_rate=dropout_rate)

        # Add transition_block
        x, nb_filter = transition_block(x, stage, nb_filter, compression=compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter,
                               growth_rate=config.growth_rate, dropout_rate=dropout_rate)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_fc = Dense(config.class_number, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet121')

    return model


def densenet121(config):
    return densenet_builder(config, [6, 12, 24, 16])


def densenet169(config):
    return densenet_builder(config, [6, 12, 32, 32])


def densenet201(config):
    return densenet_builder(config, [6, 12, 48, 32])


def densenet264(config):
    return densenet_builder(config, [6, 12, 64, 48])


