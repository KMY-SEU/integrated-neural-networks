# -*- coding: utf-8

'''
you can change the model_name from the following list:
    'AlexNet'
    'VGG16'
    'VGG19'
    'InceptionV3'
    'InceptionV4'
    'InceptionV4_ResNetV1'
    'InceptionV4_ResNetV2'
    'ResNet18'
    'ResNet34'
    'ResNet50'
    'ResNet101'
    'ResNet152'
    'DenseNet121'
    'DenseNet169'
    'DenseNet201'
    'DenseNet264'
'''


class DefaultConfig():
    model_name = 'AlexNet'

    train_data_path = './dataset/train/'
    test_data_path = './dataset/test/'
    checkpoints = './checkpoints/'

    # general params
    img_size = 224
    epochs = 100000
    batch_size = 2
    class_number = 2
    channels = 3
    lr = 0.0001
    lr_reduce_patience = 5  # 需要降低学习率的训练步长
    early_stop_patience = 10  # 提前终止训练的步长
    data_augmentation = True
    monitor = 'val_loss'
    cut = True

    # DenseNet
    growth_rate = 32
    dropout_rate = 0.2


config = DefaultConfig()
