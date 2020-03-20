# utf-8

from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers import MaxPooling2D, Conv2D, AveragePooling2D, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model

from keras import backend as K

K.set_image_data_format('channels_last')


def conv_block(x, filters, kernel_size, strides=(1, 1), padding='same', use_bias=False):
    channel_axis = -1

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=channel_axis, epsilon=0.001)(x)
    return x


def inception_stem(input):
    channel_axis = -1

    # Input Shape is 299 x 299 x 3 (tf)
    x = conv_block(input, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv_block(x, 32, (3, 3), padding='valid')
    x = conv_block(x, 64, (3, 3))

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, (3, 3), strides=(2, 2), padding='valid')

    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 64, (1, 1))
    x1 = conv_block(x1, 96, (3, 3), padding='valid')

    x2 = conv_block(x, 64, (1, 1))
    x2 = conv_block(x2, 64, (1, 7))
    x2 = conv_block(x2, 64, (7, 1))
    x2 = conv_block(x2, 96, (3, 3), padding='valid')

    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 192, (3, 3), strides=(2, 2), padding='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = concatenate([x1, x2], axis=channel_axis)
    return x


def inception_A(input):
    channel_axis = -1

    a1 = conv_block(input, 96, (1, 1))

    a2 = conv_block(input, 64, (1, 1))
    a2 = conv_block(a2, 96, (3, 3))

    a3 = conv_block(input, 64, (1, 1))
    a3 = conv_block(a3, 96, (3, 3))
    a3 = conv_block(a3, 96, (3, 3))

    a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    a4 = conv_block(a4, 96, (1, 1))

    m = concatenate([a1, a2, a3, a4], axis=channel_axis)
    return m


def inception_B(input):
    channel_axis = -1

    b1 = conv_block(input, 384, (1, 1))

    b2 = conv_block(input, 192, (1, 1))
    b2 = conv_block(b2, 224, (1, 7))
    b2 = conv_block(b2, 256, (7, 1))

    b3 = conv_block(input, 192, (1, 1))
    b3 = conv_block(b3, 192, (7, 1))
    b3 = conv_block(b3, 224, (1, 7))
    b3 = conv_block(b3, 224, (7, 1))
    b3 = conv_block(b3, 256, (1, 7))

    b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    b4 = conv_block(b4, 128, (1, 1))

    m = concatenate([b1, b2, b3, b4], axis=channel_axis)
    return m


def inception_C(input):
    channel_axis = -1

    c1 = conv_block(input, 256, (1, 1))

    c2 = conv_block(input, 384, (1, 1))
    c2_1 = conv_block(c2, 256, (1, 3))
    c2_2 = conv_block(c2, 256, (3, 1))
    c2 = concatenate([c2_1, c2_2], axis=channel_axis)

    c3 = conv_block(input, 384, (1, 1))
    c3 = conv_block(c3, 448, (3, 1))
    c3 = conv_block(c3, 512, (1, 3))
    c3_1 = conv_block(c3, 256, (1, 3))
    c3_2 = conv_block(c3, 256, (3, 1))
    c3 = concatenate([c3_1, c3_2], axis=channel_axis)

    c4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    c4 = conv_block(c4, 256, (1, 1))

    m = concatenate([c1, c2, c3, c4], axis=channel_axis)
    return m


def reduction_A(input):
    channel_axis = -1

    r1 = conv_block(input, 384, (3, 3), strides=(2, 2), padding='valid')

    r2 = conv_block(input, 192, (1, 1))
    r2 = conv_block(r2, 224, (3, 3))
    r2 = conv_block(r2, 256, (3, 3), strides=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)
    return m


def reduction_B(input):
    channel_axis = -1

    r1 = conv_block(input, 192, (1, 1))
    r1 = conv_block(r1, 192, (3, 3), strides=(2, 2), padding='valid')

    r2 = conv_block(input, 256, (1, 1))
    r2 = conv_block(r2, 256, (1, 7))
    r2 = conv_block(r2, 320, (7, 1))
    r2 = conv_block(r2, 320, (3, 3), strides=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)
    return m


def inception_v4(config):
    print('keras backend: ', K.backend())

    input_shape = (config.img_size, config.img_size, config.channels)
    img_input = Input(shape=input_shape)

    # Input Shape is 299 x 299 x 3 (tf)
    x = inception_stem(img_input)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)

    # Full Connection
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.2)(x)

    # Output
    out = Dense(config.class_number, activation='softmax')(x)

    model = Model(img_input, out, name='Inception_v4')

    return model
