from keras.layers import Input, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers import MaxPooling2D, Conv2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate, add
from keras.models import Model

import warnings
from keras import backend as K

warnings.filterwarnings('ignore')
K.set_image_data_format('channels_last')


def inception_resnet_stem(input):
    channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    c = Conv2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    c = Conv2D(32, 3, 3, activation='relu', )(c)
    c = Conv2D(64, 3, 3, activation='relu', border_mode='same')(c)

    c1 = MaxPooling2D((3, 3), strides=(2, 2))(c)
    c2 = Conv2D(96, 3, 3, activation='relu', subsample=(2, 2))(c)

    m = concatenate([c1, c2], axis=channel_axis)

    c1 = Conv2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c1 = Conv2D(96, 3, 3, activation='relu', )(c1)

    c2 = Conv2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c2 = Conv2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    c2 = Conv2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    c2 = Conv2D(96, 3, 3, activation='relu', border_mode='valid')(c2)

    m2 = concatenate([c1, c2], axis=channel_axis)

    p1 = MaxPooling2D((3, 3), strides=(2, 2), )(m2)
    p2 = Conv2D(192, 3, 3, activation='relu', subsample=(2, 2))(m2)

    m3 = concatenate([p1, p2], axis=channel_axis)
    m3 = BatchNormalization(axis=channel_axis)(m3)
    m3 = Activation('relu')(m3)
    return m3


def inception_resnet_v2_A(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(ir2)

    ir3 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir3 = Conv2D(48, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(ir3)

    ir_merge = concatenate([ir1, ir2, ir3], axis=channel_axis)

    ir_conv = Conv2D(384, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def inception_resnet_v2_B(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Conv2D(128, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Conv2D(160, 1, 7, activation='relu', border_mode='same')(ir2)
    ir2 = Conv2D(192, 7, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(1152, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def inception_resnet_v2_C(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Conv2D(192, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Conv2D(224, 1, 3, activation='relu', border_mode='same')(ir2)
    ir2 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(2144, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = BatchNormalization(axis=channel_axis)(out)
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
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m


def reduction_resnet_v2_B(input):
    channel_axis = -1

    r1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    r2 = Conv2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Conv2D(384, 3, 3, activation='relu', subsample=(2, 2))(r2)

    r3 = Conv2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Conv2D(288, 3, 3, activation='relu', subsample=(2, 2))(r3)

    r4 = Conv2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r4 = Conv2D(288, 3, 3, activation='relu', border_mode='same')(r4)
    r4 = Conv2D(320, 3, 3, activation='relu', subsample=(2, 2))(r4)

    m = concatenate([r1, r2, r3, r4], axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m


def inception_resnet_v2(config):
    scale = True

    input_shape = (config.img_size, config.img_size, config.channels)
    init = Input(shape=input_shape)

    # Input Shape is 299 x 299 x 3 (tf)
    x = inception_resnet_stem(init)

    # 10 x Inception Resnet A
    for i in range(10):
        x = inception_resnet_v2_A(x, scale_residual=scale)

    # Reduction A
    x = reduction_A(x, k=256, l=256, m=384, n=384)

    # 20 x Inception Resnet B
    for i in range(20):
        x = inception_resnet_v2_B(x, scale_residual=scale)

    # Reduction Resnet B
    x = reduction_resnet_v2_B(x)

    # 10 x Inception Resnet C
    for i in range(10):
        x = inception_resnet_v2_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=config.class_number, activation='softmax')(x)

    model = Model(init, out, name='Inception-Resnet-v2')

    return model
