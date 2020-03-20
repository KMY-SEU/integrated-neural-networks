# utf-8

from __future__ import print_function
from config import config
import numpy as np
import os, sys, glob, itertools, tqdm, cv2, keras
import tensorflow as tf

from random import shuffle
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from model_factory import Model
from keras.backend.tensorflow_backend import set_session

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.setrecursionlimit(10000)

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
set_session(tf.Session(config=config1))


class Train(Model):
    def __init__(self, config):
        Model.__init__(self, config)

    def get_file(self, path):
        ends = os.listdir(path)[0].split('.')[-1]
        img_list = glob.glob(os.path.join(path, '*.' + ends))
        return img_list

    # load data
    def load_data(self):
        categories = list(map(self.get_file, list(
            map(lambda x: os.path.join(self.train_data_path, x), os.listdir(self.train_data_path)))))
        data_list = list(itertools.chain.from_iterable(categories))
        shuffle(data_list)
        images_data, labels_idx, labels = [], [], []

        with_platform = os.name

        print('data loading...')
        for i in tqdm.trange(len(data_list), ascii=True):
            file = data_list[i]
            img = cv2.imread(file)

            try:
                _, w, h = img.shape[::-1]
            except:
                print(file)

            # crop image's margin
            if self.cut:
                # crop centrally
                if h <= w:
                    delta = w - h
                    x_offset = delta // 2
                    img = img[:, x_offset: w + x_offset - delta, :]
                else:
                    delta = h - w
                    y_offset = delta // 2
                    img = img[y_offset: h + y_offset - delta, :, :]

            # resize image to determined img_size
            img = cv2.resize(img, (self.img_size, self.img_size))

            if with_platform == 'posix':
                label = file.split('/')[-2][-1]
            elif with_platform == 'nt':
                label = file.split('\\')[-2][-1]

            img = img_to_array(img)
            images_data.append(img)
            labels.append(label)

        with open('train_class_idx.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            for label in labels:
                idx = lines.index(label.rstrip())
                labels_idx.append(idx)

        images_data = np.array(images_data, dtype=np.float64) / 255.0
        labels = to_categorical(np.array(labels_idx), num_classes=self.class_number)
        X_train, X_test, y_train, y_test = train_test_split(images_data, labels)

        return X_train, X_test, y_train, y_test

    def mkdir(self, path):
        if not os.path.exists(path):
            return os.mkdir(path)
        return path

    def train(self, X_train, X_test, y_train, y_test, model):
        tensorboard = TensorBoard(log_dir=self.mkdir(os.path.join(self.checkpoints, self.model_name)))

        lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor=config.monitor,
                                                      factor=0.1,
                                                      patience=config.lr_reduce_patience,
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)

        early_stop = keras.callbacks.EarlyStopping(monitor=config.monitor,
                                                   min_delta=0,
                                                   patience=config.early_stop_patience,
                                                   verbose=1,
                                                   mode='auto')

        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(self.mkdir(os.path.join(self.checkpoints, self.model_name)), self.model_name + '.h5'),
            monitor=config.monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1)

        # data augmentation
        if self.data_augmentation:
            print("using data augmentation method")

            data_aug = ImageDataGenerator(
                rotation_range=10,  # 图像旋转的角度
                width_shift_range=0.2,  # 左右平移参数
                height_shift_range=0.2,  # 上下平移参数
                zoom_range=0.2,  # 随机放大或者缩小
                horizontal_flip=True,  # 随机翻转
            )

            data_aug.fit(X_train)

            print('the current model is ', config.model_name)
            model.fit_generator(
                generator=data_aug.flow(X_train, y_train, batch_size=config.batch_size),
                steps_per_epoch=X_train.shape[0] // self.batch_size,
                validation_data=(X_test, y_test),
                shuffle=True,
                epochs=self.epochs, verbose=1, max_queue_size=100,
                callbacks=[early_stop, checkpoint, lr_reduce, tensorboard],
            )
        else:
            print('the current model is ', config.model_name)
            model.fit(x=X_train, y=y_train,
                      batch_size=self.batch_size,
                      validation_data=(X_test, y_test),
                      epochs=self.epochs,
                      callbacks=[early_stop, lr_reduce, tensorboard],
                      shuffle=True,
                      verbose=1)

    def start_train(self):
        # load data
        X_train, X_test, y_train, y_test = self.load_data()
        # create model
        model = Model(config).create_model()
        # train model
        self.train(X_train, X_test, y_train, y_test, model)

    def remove_logdir(self):
        self.mkdir(self.checkpoints)
        self.mkdir(os.path.join(self.checkpoints, self.model_name))
        events = os.listdir(os.path.join(self.checkpoints, self.model_name))
        for evs in events:
            if "events" in evs:
                os.remove(os.path.join(os.path.join(self.checkpoints, self.model_name), evs))

    def mkdir(self, path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path


def main():
    train = Train(config)
    train.remove_logdir()
    train.start_train()
    print('Done')


if __name__ == '__main__':
    main()
