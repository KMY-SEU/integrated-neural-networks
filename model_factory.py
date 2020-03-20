# utf-8

from __future__ import print_function
import keras

from alexnet import alexnet
from vgg16 import vgg16
from vgg19 import vgg19
from inception_v3 import inception_v3
from inception_v4 import inception_v4
from inception_resnet_v1 import inception_resnet_v1
from inception_resnet_v2 import inception_resnet_v2
from resnet import ResnetBuilder
from densenet import *

import sys

sys.setrecursionlimit(10000)


class Model(object):
    def __init__(self, config):
        self.train_data_path = config.train_data_path
        self.checkpoints = config.checkpoints
        self.img_size = config.img_size
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.class_number = config.class_number
        self.model_name = config.model_name
        self.lr = config.lr
        self.config = config
        self.data_augmentation = config.data_augmentation
        self.cut = config.cut

    def model_confirm(self, choosed_model):
        if choosed_model == 'AlexNet':
            model = alexnet(self.config)
        elif choosed_model == 'VGG16':
            model = vgg16(self.config)
        elif choosed_model == 'VGG19':
            model = vgg19(self.config)
        elif choosed_model == 'InceptionV3':
            model = inception_v3(self.config)
        elif choosed_model == 'InceptionV4':
            model = inception_v4(self.config)
        elif choosed_model == 'InceptionV4_ResNetV1':
            model = inception_resnet_v1(self.config)
        elif choosed_model == 'InceptionV4_ResNetV2':
            model = inception_resnet_v2(self.config)
        elif choosed_model == 'ResNet18':
            model = ResnetBuilder.build_resnet_18(self.config)
        elif choosed_model == 'ResNet34':
            model = ResnetBuilder.build_resnet_34(self.config)
        elif choosed_model == 'ResNet50':
            model = ResnetBuilder.build_resnet_50(self.config)
        elif choosed_model == 'ResNet101':
            model = ResnetBuilder.build_resnet_101(self.config)
        elif choosed_model == 'ResNet152':
            model = ResnetBuilder.build_resnet_152(self.config)
        elif choosed_model == 'DenseNet121':
            model = densenet121(self.config)
        elif choosed_model == 'DenseNet169':
            model = densenet169(self.config)
        elif choosed_model == 'DenseNet201':
            model = densenet201(self.config)
        elif choosed_model == 'DenseNet264':
            model = densenet264(self.config)
        else:
            model = -1

        return model

    def model_compile(self, model):
        adam = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])  # compile之后才会更新权重和模型
        return model

    def create_model(self):
        model = self.model_confirm(self.model_name)
        model = self.model_compile(model)
        return model
