# utf-8

from __future__ import division

from keras.layers import Flatten

from keras.models import Sequential

from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras import backend as K
K.set_image_data_format('channels_last')

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations


def alexnet(config):
    model = Sequential()
    input_shape = (config.img_size, config.img_size, config.channels)

    model.add(Conv2D(96, (11, 11), input_shape=input_shape, strides=(4, 4),
                     padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.class_number, activation='softmax'))

    return model
