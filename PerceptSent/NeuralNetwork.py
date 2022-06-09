import os
import cv2
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, Xception, DenseNet169
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

BATCH_SIZE = 8
VERBOSE = 1

class DataGenerator(Sequence):
    
    def __init__(self, img_files,
                 labels,
                 att_dir=None,
                 img_size=None,
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 permutation=None,
                 nr_classes=3):

        self.img_files = img_files
        self.att_dir = att_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.permutation = permutation
        self.labels = labels
        self.nr_classes = nr_classes
        self.__on_epoch_end()

    def __len__(self):
        return int((len(self.img_files)-1)/self.batch_size+1)

    def __getitem__(self, index):
        if index == self.__len__():
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        img_files_temp = [self.img_files[k] for k in indexes]
        if self.labels is not None:
            labels_temp = [self.labels[k] for k in indexes]
            X, y = self.__data_generation(img_files_temp, labels_temp)
            return X, y
        else:
            X = self.__data_generation(img_files_temp)
            return X

    def __on_epoch_end(self):
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __crop_resize(self, img_file):
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        y, x, c = img.shape
        if y < x:
            aux = int((x-y)/2)
            img = img[:,aux:aux+y]
        elif x < y:
            aux = int((y-x)/2)
            img = img[aux:aux+x,:]
        return np.float32(cv2.resize(img, (self.img_size, self.img_size)))#/255.
        
    def __data_generation(self, img_files_temp, labels_temp=None):
        X_img = list()
        X_att = list()
        Y = list()

        if self.labels is None:
            element_size = len(img_files_temp[0])
            labels_temp = [[None]*element_size for _ in range(len(img_files_temp))]
        
        for img, label in zip(img_files_temp, labels_temp):
            img_data = self.__crop_resize(img)
            X_img.append(img_data)
            if self.att_dir is not None:
                img_name = ((img.split('/')[-1]).split('.')[0])
                a = f"{self.att_dir}/{img_name}.txt"
                att = np.loadtxt(a)
                X_att.append(att)
            Y.append(label)
        
        if self.permutation is not None:
            for p in self.permutation:
                X_att[0][p] = -1

        if self.att_dir is not None:
            X = [np.asarray(X_img), np.asarray(X_att)]
        else:
            X = np.asarray(X_img)
        if self.labels is None:
            return X
        else:
            return X, to_categorical(Y, num_classes=self.nr_classes)

class NeuralNetwork:

    __known_models = ['none', 'inception', 'resnet', 'vgg',
                      'xception', 'densenet', 'robust']

    def __init__(self, dataset, freeze=False, early_stop=False, h5_file=None):   
        self.freeze = freeze
        self.early_stop = early_stop
        self.dataset = dataset
        self.results_folder = dataset.create_dir(f"{dataset.output_folder}/results")
        assert(dataset.model in self.__known_models)
        self.model_name = dataset.model
        if self.model_name == 'inception':
            self.img_size = 299
        else:
            self.img_size = 224
        self.model = self.__load_model()
        if h5_file:
            # Load trained model.
            self.model.load_weights(h5_file)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def add_dense_layers(self, input_tensor, x=None):
        if self.model_name != 'none':
            if self.dataset.attributes is not None:
                    att_tensor = layers.Input(shape=(self.dataset.n_atts,), name='attributes')
                    x = layers.Concatenate()([x, att_tensor])
        else:
            if self.dataset.attributes is not None:
                    att_tensor = layers.Input(shape=(self.dataset.n_atts,), name='attributes')
                    x = att_tensor
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(24, activation='relu')(x)
        x = layers.Dense(self.dataset.nr_classes, activation='softmax')(x)

        if self.dataset.attributes is not None:
            model = Model(inputs=[input_tensor, att_tensor], outputs=x)
        else:
            model = Model(inputs=[input_tensor], outputs=x)
        return model

    def __load_model(self):

        def robust_model(input_tensor):
            # Robust Image Sentiment Analysis Using
            #  Progressively Trained and Domain Transferred Deep Networks
            # https://arxiv.org/abs/1509.06041
            x = layers.Conv2D(filters=96,
                            kernel_size=(11,11),
                            strides=4,
                            activation='relu')(input_tensor)
            x = layers.Lambda(lambda a: tf.nn.lrn(input=a))(x)
            x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)

            x = layers.Conv2D(filters=256,
                                kernel_size=(5, 5),
                                strides=2,
                                activation='relu',
                                name = "conv2d_last")(x)
            x = layers.Lambda(lambda a: tf.nn.lrn(input=a))(x)
            x = layers.MaxPooling2D(name="max_pooling2d_lcl1", pool_size=(6, 6), strides=6)(x)

            return Model(inputs=input_tensor, outputs=x)

        input_tensor = layers.Input(shape=(self.img_size, self.img_size, 3), name='images')

        if self.model_name == 'robust':
            last_conv_layer_name = "conv2d_last"
            base_model = robust_model(input_tensor)
        elif self.model_name == 'vgg':
            # Very Deep Convolutional Networks
            #  for Large-Scale Image Recognition
            # https://arxiv.org/abs/1409.1556
            preprocessed_input = vgg16_preprocess_input(input_tensor)
            last_conv_layer_name = "block5_pool"
            base_model = VGG16(
                               weights="imagenet",
                               include_top=False,
                               input_tensor=preprocessed_input
                              )
        elif self.model_name == 'resnet':
            # Deep Residual Learning for Image Recognition
            # https://arxiv.org/abs/1512.03385
            preprocessed_input = resnet_preprocess_input(input_tensor)
            last_conv_layer_name = "conv5_block3_out"
            base_model = ResNet50(
                                  weights='imagenet',
                                  include_top=False,
                                  input_tensor=preprocessed_input
                                 )
        elif self.model_name == 'inception':
            # Rethinking the Inception Architecture for Computer Vision
            # https://arxiv.org/abs/1512.00567
            preprocessed_input = inception_preprocess_input(input_tensor)
            last_conv_layer_name = "mixed10"
            base_model = InceptionV3(
                                     weights="imagenet",
                                     include_top=False,
                                     input_tensor=preprocessed_input
                                    )
        elif self.model_name == 'xception':
            # Xception: Deep Learning with Depthwise Separable Convolutions
            # https://arxiv.org/abs/1610.02357
            preprocessed_input = xception_preprocess_input(input_tensor)
            last_conv_layer_name = "block14_sepconv2_act"
            base_model = Xception(                                  
                                  weights="imagenet",
                                  include_top=False,
                                  input_tensor=preprocessed_input
                                 )
        elif self.model_name == 'densenet':
            # Densely Connected Convolutional Networks
            # https://arxiv.org/abs/1608.06993
            preprocessed_input = densenet_preprocess_input(input_tensor)
            last_conv_layer_name = "relu"
            base_model = DenseNet169(                                     
                                     weights="imagenet",
                                     include_top=False,
                                     input_tensor=preprocessed_input
                                    )

        if self.model_name != 'none':
            self.last_conv_layer = base_model.get_layer(last_conv_layer_name)

            if self.freeze:
                base_model.trainable = False
                next_layer = base_model(input_tensor, training=False)
            else:
                next_layer = base_model.output

            if self.model_name not in ['vgg', 'robust']:
                x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(next_layer)
                self.classifier_layer_names = ["global_average_pooling2d"]
            else:
                x = layers.Flatten(name="flatten")(next_layer)
                self.classifier_layer_names = ["flatten"]

            model = self.add_dense_layers(input_tensor, x)
        else:
            model = self.add_dense_layers(input_tensor)
        
        return model

    def load_dataset(self, classify=False):

        if not self.dataset.images:
            raise Exception("Image folder not informed.")         

        if classify:
            raise Exception("Not implemented")
        
        X_imgs = list()
        Y_labels = list()
        assert(os.path.isfile(self.dataset.dataset_file))
        with open(self.dataset.dataset_file) as f:
            for line in f:
                img_data = line.replace("\n","").split(" ")
                imgs = list()
                labels = list()
                if len(img_data) == 2:
                    imgs.append([self.dataset.images, self.dataset.images_data_augmented, img_data[0]])
                    labels.append(int(img_data[1]))
                elif len(img_data) == 11:
                    for i in range (1,10,2):
                        imgs.append([self.dataset.images, self.dataset.images_data_augmented, f"{img_data[0]}_{img_data[i]}"])
                        labels.append(int(img_data[i+1]))
                else: 
                    raise Exception("Erro while reading image data from dataset.")
                X_imgs.append(imgs)
                Y_labels.append(labels)
        X = np.asarray(X_imgs)
        Y = np.asarray(Y_labels)
        self.__unpack(X, Y, "imgs")
        return X, Y

    def train_model(self, X_imgs, Y_labels, epochs=20, k=5, permutation=False):

        def save_model(file_name):
            with open(file_name, 'a+') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        w_folder = f"{self.results_folder}/Weights"
        self.dataset.create_dir(w_folder)
        save_model(f"{self.results_folder}/model.txt")
        assert(k > 0 and k < 10)
        y = [sum(y) for y in Y_labels]
        if k > 1:
            kf = StratifiedKFold(n_splits=k)
            self.model.save_weights(f'{w_folder}/temp_weights.h5')            
            for i, (tr, te) in enumerate(kf.split(np.zeros(len(y)), y)):
                if i != 0:
                    self.model.load_weights(f'{w_folder}/temp_weights.h5')
                X_train = list()
                Y_train = list()
                X_val = list()
                Y_val = list()
                for k in tr:
                    X_train.append(X_imgs[k])
                    Y_train.append(Y_labels[k])
                for k in te:
                    X_val.append(X_imgs[k])
                    Y_val.append(Y_labels[k])
                self.__fit_model(X_train, Y_train, epochs, early_stop=self.early_stop, w_folder=w_folder, name=f"{k}_{i+1}")
                self.classify(X_val, Y_val, f"{k}_{i+1}")

                if permutation:
                    for p in range(0,self.dataset.n_atts):
                        self.classify(X_val, Y_val, name=f'{k}_{i+1}_perm{p}', permutation=[p])

        elif k == 1:
            X_train, X_val, Y_train, Y_val = train_test_split(X_imgs, Y_labels, test_size = 0.2, stratify = y, random_state = 42)
            self.__fit_model(X_train, Y_train, epochs=epochs, early_stop=self.early_stop, w_folder=w_folder, name=f"{k}_1")
            self.classify(X_val, Y_val, f"{k}_1")   

            if permutation:
                for p in range(0,self.dataset.n_atts):
                    self.classify(X_val, Y_val, name=f'{k}_1_perm{p}', permutation=[p])             

    def __unpack(self, X, Y, file_name=None, classify=False):
        XY = list()

        for imgs, labels in zip(X, Y):
            if classify:
                for img, label in zip(imgs, labels):
                    img_path = f"{img[0]}/{img[2]}"
                    if ".jpg" not in img_path:
                        img_path = f"{img_path}.jpg"
                    XY.append([img_path, int(label)])
            else:
                for img, label in zip(imgs, labels):
                    if self.dataset.data_augmented:
                        img_path = f"{img[1]}/{img[2]}"
                    else:
                        img_path = f"{img[0]}/{img[2]}"
                    if ".jpg" not in img_path:
                        img_path = f"{img_path}.jpg"
                    XY.append([img_path, int(label)])
        if file_name:
            with open(f"{self.results_folder}/{file_name}.txt", 'w') as f:
                for x, y in XY:
                    f.write(f"{str(x)} {str(y)}\n")
        X_imgs = list()
        Y_labels = list()
        for x, y in XY:
            X_imgs.append(x)
            Y_labels.append(y)        
        return X_imgs, Y_labels

    def __fit_model(self, X_imgs, Y_labels, epochs=20, early_stop=False, w_folder="output/Weights", name="model"):


        def lr_scheduler(self, epoch):
            if epoch >= 0.7*epochs:
                return 1e-6
            elif epoch >= 0.4*epochs:
                return 1e-5
            else:
                return 1e-4

        def weights(y):
            c = np.zeros(self.dataset.nr_classes)
            for i in range(self.dataset.nr_classes):
                c[i] = np.sum(np.where(y==i, 1., 0.))

            class_weight = {}
            for i in range(self.dataset.nr_classes):
                class_weight[i] = min(c)/c[i]
            print('Class Weights:', class_weight)
            return class_weight

        y = [sum(y) for y in Y_labels]
        X_train, X_val, Y_train, Y_val = train_test_split(X_imgs, Y_labels, stratify = y, test_size = 0.2, random_state = 42)
        Xt, Yt = self.__unpack(X_train, Y_train, f"imgs_train_{name}")
        Xv, Yv = self.__unpack(X_val, Y_val, f"imgs_val_{name}")
        class_weight = weights(np.asarray(Yt))
        train_datagen = DataGenerator(Xt, Yt,
                                      att_dir = self.dataset.attributes,
                                      img_size = self.img_size,
                                      batch_size = BATCH_SIZE,
                                      nr_classes=self.dataset.nr_classes)
        val_datagen = DataGenerator(Xv, Yv,
                                    att_dir = self.dataset.attributes,
                                    img_size = self.img_size,
                                    batch_size = BATCH_SIZE,                                    
                                    nr_classes=self.dataset.nr_classes)

        h5_file = f'{w_folder}/weights_{name}.h5'

        checkpoint = ModelCheckpoint(h5_file,
                                     monitor = 'val_accuracy',
                                     verbose = 1,
                                     save_best_only = True,
                                     save_weights_only = True,
                                     save_freq = "epoch",
                                     mode = 'max'
                                    )
        lr_decay = LearningRateScheduler(lr_scheduler)

        history_logger=tf.keras.callbacks.CSVLogger(f"{w_folder}/history_{name}.csv", separator=",", append=True)

        if early_stop:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                                                    monitor='val_accuracy', min_delta=0.001, patience=10, verbose=VERBOSE,
                                                    mode='auto', baseline=None, restore_best_weights=True
                                                    )
            callbacks = [lr_decay, checkpoint, early_stopping, history_logger]
        else:
            callbacks = [lr_decay, checkpoint, history_logger]

        self.model.fit(train_datagen,
                        epochs = epochs,
                        validation_data = val_datagen,
                        callbacks=callbacks,
                        verbose=VERBOSE,
                        class_weight = class_weight)
        return h5_file

    def classify(self, X_imgs, Y_labels, name="classify", unpack=True, permutation=None):
        
        if unpack:
            imgs, y_true = self.__unpack(X_imgs, Y_labels, name, True)
        else:
            imgs = X_imgs
            y_true = Y_labels
        test_datagen = DataGenerator(imgs, None,
                                     att_dir = self.dataset.attributes,
                                     img_size = self.img_size,
                                     batch_size = 1,
                                     shuffle = False,
                                     permutation=permutation,
                                     nr_classes=self.dataset.nr_classes)
        pred = self.model.predict(test_datagen, verbose=VERBOSE)
        y_pred = np.argmax(pred, axis = 1)
        if y_true is not None:
            with open(f"{self.results_folder}/test_{name}.txt", 'a+') as f:
                f.write(str(classification_report(y_true, y_pred)))
                f.write("\n")
                f.write(str(accuracy_score(y_true, y_pred)))
                f.write("\n")
                f.write(str(confusion_matrix(y_true, y_pred)))
                f.write("\n")
                classification = list()
                f.write("image, value, true_label, predict_label\n")
                for img, p, c, t in zip(imgs, pred, y_pred, y_true):
                    f.write(f"{img}, {p}, {self.dataset.classes_list[t]}, {self.dataset.classes_list[c]}\n")
                    classification.append([img, p[0], p[1], self.dataset.classes_list[t], self.dataset.classes_list[c]])
        return classification
