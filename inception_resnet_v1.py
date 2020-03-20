from keras.layers import Input, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers import MaxPooling2D, Conv2D, AveragePooling2D, BatchNormalization
from keras.layers.merge import concatenate, add
from keras.models import Model

from keras import backend as K

K.set_image_data_format('channels_last')

import warnings

warnings.filterwarnings('ignore')


def inception_resnet_stem(input):
    channel_axis = -1

    # Input Shape is 299 x 299 x 3 (tf)
    c = Conv2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    c = Conv2D(32, 3, 3, activation='relu', )(c)
    c = Conv2D(64, 3, 3, activation='relu', )(c)
    c = MaxPooling2D((3, 3), strides=(2, 2))(c)
    c = Conv2D(80, 1, 1, activation='relu', border_mode='same')(c)
    c = Conv2D(192, 3, 3, activation='relu')(c)
    c = Conv2D(256, 3, 3, activation='relu', subsample=(2, 2), border_mode='same')(c)
    b = BatchNormalization(axis=channel_axis, epsilon=0.001)(c)
    b = Activation('relu')(b)
    return b


def inception_resnet_A(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(ir2)

    ir3 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir3 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(ir3)

    ir_concatenate = concatenate([ir1, ir2, ir3], axis=channel_axis)

    ir_conv = Conv2D(256, 1, 1, activation='linear', border_mode='same')(ir_concatenate)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis, epsilon=0.001)(out)
    out = Activation("relu")(out)
    return out


def inception_resnet_B(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(128, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Conv2D(128, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Conv2D(128, 1, 7, activation='relu', border_mode='same')(ir2)
    ir2 = Conv2D(128, 7, 1, activation='relu', border_mode='same')(ir2)

    ir_concatenate = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(896, 1, 1, activation='linear', border_mode='same')(ir_concatenate)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis, epsilon=0.001)(out)
    out = Activation("relu")(out)
    return out


def inception_resnet_C(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(128, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Conv2D(192, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Conv2D(192, 1, 3, activation='relu', border_mode='same')(ir2)
    ir2 = Conv2D(192, 3, 1, activation='relu', border_mode='same')(ir2)

    ir_concatenate = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(1792, 1, 1, activation='linear', border_mode='same')(ir_concatenate)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis, epsilon=0.001)(out)
    out = Activation("relu")(out)
    return out


def reduction_A(input, k=192, l=224, m=256, n=384):
    channel_axis = -1

    r1 = MaxPooling2D((3, 3), strides=(2, 2))(input)

    r2 = Conv2D(n, 3, 3, activation='relu', subsample=(2, 2))(input)

    r3 = Conv2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Conv2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Conv2D(m, 3, 3, activation='relu', subsample=(2, 2))(r3)

    m = concatenate([r1, r2, r3], axis=channel_axis)
    m = BatchNormalization(axis=channel_axis, epsilon=0.001)(m)
    m = Activation('relu')(m)
    return m


def reduction_resnet_B(input):
    channel_axis = -1

    r1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    r2 = Conv2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Conv2D(384, 3, 3, activation='relu', subsample=(2, 2))(r2)

    r3 = Conv2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Conv2D(256, 3, 3, activation='relu', subsample=(2, 2))(r3)

    r4 = Conv2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(r4)
    r4 = Conv2D(256, 3, 3, activation='relu', subsample=(2, 2))(r4)

    m = concatenate([r1, r2, r3, r4], axis=channel_axis)
    m = BatchNormalization(axis=channel_axis, epsilon=0.001)(m)
    m = Activation('relu')(m)
    return m


def inception_resnet_v1(config):
    scale = True

    input_shape = (config.img_size, config.img_size, config.channels)
    init = Input(shape=input_shape)

    # Input Shape is 299 x 299 x 3 (tf)
    x = inception_resnet_stem(init)

    # 5 x Inception Resnet A
    for i in range(5):
        x = inception_resnet_A(x, scale_residual=scale)

    # Reduction A - From Inception v4
    x = reduction_A(x, k=192, l=192, m=256, n=384)

    # 10 x Inception Resnet B
    for i in range(10):
        x = inception_resnet_B(x, scale_residual=scale)

    # Reduction Resnet B
    x = reduction_resnet_B(x)

    # 5 x Inception Resnet C
    for i in range(5):
        x = inception_resnet_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=config.class_number, activation='softmax')(x)

    model = Model(init, out, name='Inception-Resnet-v1')

    return model
