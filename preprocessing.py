import pathlib
import tensorflow as tf
from csv import reader
import numpy as np
import os
import operator
import math
import matplotlib.pyplot as plt
from datetime import datetime
import time
import csv

class Preprocessing:
    def __init__(self,train=False):
        self.file = pathlib.Path('.\\cs-t0828-2020-hw1\\')  # windows
        # self.filepath = pathlib.Path('') # ubuntu

        self.N_batch = 16

        self.resolution = 50 #224 # resize/crop to square image of this size


        self.N_classes = 196
        self.N_train = 0
        self.N_test = 0
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.file_list_train = []
        self.file_list_test = []
        self.train_labels = [] # list of integers corresponding to label string
        self.unique_labels = [] # 196 label strings

        self.preprocess()

        self.ds_input = self.create_dataset_input()
        self.ds_target = self.create_dataset_target()
        self.ds_input_target = self.create_dataset_input_target()
        self.ds_input_test = self.create_dataset_input_test()

        self.model = self.build_model()

        if train:
            logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=logs,
                histogram_freq=1
            )
            history = self.model.fit(
                self.ds_input_target,
                batch_size=self.N_batch,
                epochs=5,
                #steps_per_epoch=self.N_train,
                callbacks=[tboard_callback]
            )
            self.model.save('./models/model/')
        return

    def preprocess(self):
        '''
        convert from label strings to one-hot vectors

        for example,
        'Ford F-150 Regular Cab 2007' --> 0
        'BMW X6 SUV 2012' --> 1
        ...
        'BMW 1 Series Coupe 2012' --> 195
        '''

        # ------------------- notes -------------------
        # y_train = tf.keras.utils.to_categorical(y_train, self.N_classes)
        # y_test = tf.keras.utils.to_categorical(y_test, self.N_classes)

        unique_labels = []
        #y_train = np.array(0)
        #y_test = np.array(0)
        # for row in .csv
        # if column[1] in unique_labels:
        #     set train/test label value
        # else:
        #     unique_labels.append(column[1])
        #     set train/test label value


        # first, scan train/test directories and count number of examples
        train_path = os.path.join(self.file,'training_data','training_data')
        test_path = os.path.join(self.file, 'testing_data', 'testing_data')
        csv = self.file.joinpath('training_labels.csv')

        # scan each dir and count number of examples
        train_contents = sorted(os.listdir(train_path)) # list
        test_contents = sorted(os.listdir(test_path)) # list
        r = reader(open(csv))
        next(r) # skip header
        csv_sorted = sorted(r, key=operator.itemgetter(0)) # list
        self.N_train = len(train_contents)
        self.N_test = len(test_contents)
        print('found', self.N_train, 'training datas') # 11185
        print('found', self.N_test, 'testing datas') # 5000
        print(train_contents[0], train_contents[1]) # 000001.jpg 000002.jpg
        print(csv_sorted[0], csv_sorted[1]) # ['000001', 'AM General Hummer SUV 2000'] ['000002', 'AM General Hummer SUV 2000']

        # now that image paths (train/test_contents) and labels (csv_sorted) are sorted, save to self
        # then use tf.data with generator
        train_contents_full = []
        test_contents_full = []
        for obj in train_contents:
            train_contents_full.append(os.path.join(train_path,obj))
        for obj in test_contents:
            test_contents_full.append(os.path.join(test_path,obj))
        self.file_list_train = train_contents_full
        self.file_list_test = test_contents_full
        #self.train_labels = csv_sorted


        # convert labels to one-hot vectors
        unique_labels = [] # list of label strings; ex. ['AM General Hummer SUV 2000', ...]
        id_counter = -1 # count number of unique labels so far
        id_list = [] # scroll csv_sorted and save list of class integers for one-hot encoding; use with tf.keras.utils.to_categorical()
        for row in csv_sorted:
            id = row[0]
            label_string = row[1]
            if label_string in unique_labels:
                idx = unique_labels.index(label_string)
                id_list.append(idx)
            else:
                unique_labels.append(label_string)
                id_counter += 1
                id_list.append(id_counter)

        print(len(id_list)) # self.N_train
        print(id_list)
        print(len(unique_labels)) # 196
        print(unique_labels)

        self.train_labels = np.array(id_list,dtype=int)
        self.unique_labels = unique_labels

        # to get the label string for each training example:
        # for row in csv_sorted:
        #     id = row[0]
        #     print(self.get_label(id))

        # during target generator (counter k), do like tf.keras.utils.to_categorical(data,num_classes)
        # out = np.zeros(N_classes,dtype=np.float32), out[id_list[k]] = 1
        return

    def get_label(self,id):
        '''
        return label string corresponding to integer id for one-hot encoding
        '''
        return self.unique_labels[id]

    def create_dataset_input(self):
        ds_input = tf.data.Dataset.from_generator(
            self.generator_input,
            args=(self.resolution, self.file_list_train),
            output_types=(tf.float32),
            output_shapes=(tf.TensorShape([self.resolution,self.resolution,3]))
        )
        ds_input = ds_input.batch(self.N_batch)
        return ds_input

    def create_dataset_input_test(self):
        ds_input = tf.data.Dataset.from_generator(
            self.generator_input,
            args=(self.resolution, self.file_list_test),
            output_types=(tf.float32),
            output_shapes=(tf.TensorShape([self.resolution, self.resolution, 3]))
        )
        ds_input = ds_input.batch(self.N_batch)
        return ds_input


    @staticmethod
    def generator_input(resolution, file_list):
        N_images = len(file_list)
        count = 0
        while count<N_images:
            if (count >= N_images):
                count = 0

            # resize and crop (center) image to (resolution,resolution,3)

            # get image
            img = tf.io.read_file(file_list[count])
            img = tf.image.decode_jpeg(img,3) # return RGB image # aka tf.io.decode_png
            img = tf.image.convert_image_dtype(img,tf.float32) # TODO: why does this return values like 1.0000002 ?  ignore for now
            #img = img.numpy()

            ## get dimensions
            #img_shape = img.shape
#
            ## resize to minimum dimension
            #min_res = 150 # px
            #if img_shape[0] < min_res:
            #    resize_factor = min_res/img_shape[0]
            #elif img_shape[1] < min_res:
            #    resize_factor = min_res/img_shape[1]
            #else:
            #    resize_factor = 1.0
#
            ## fix the logic here; find smallest dimension; resize using it
            #min_dim = np.min([img_shape[0],img_shape[1]])
            #resize_factor = min_res/min_dim
#
#
#
            ## resize to at least min_res; keep aspect ratio
            #img = tf.image.resize(img, (math.ceil(img_shape[0]*resize_factor), math.ceil(img_shape[1]*resize_factor)), antialias=True)
#
            ## random crop a square
            #output_size = 130 # px
            #img = tf.image.random_crop(img, [output_size,output_size,3])  # should be smaller than min_res

            # optionally, also change image brightness, saturation, hue, etc.
            # see https://www.tensorflow.org/tutorials/images/data_augmentation

            # the most important thing to identify a car is the logo; if crop doesn't see it, probably fail
            # therefore, use resize_with_pad()
            #img = tf.image.resize_with_pad(img,resolution,resolution,antialias=True)
            img = tf.image.resize(img,[resolution,resolution],antialias=True)

            count += 1
            yield img

    def create_dataset_target(self):
        ds_target = tf.data.Dataset.from_generator(
            self.generator_target,
            args=(self.train_labels,self.N_classes),
            output_types=(tf.float32),
            output_shapes=(
                tf.TensorShape([self.N_classes])
            )
        )
        ds_target = ds_target.batch(self.N_batch)
        return ds_target

    @staticmethod
    def generator_target(id_list,N_classes):
        '''
        id_list = self.train_labels

        '''
        N_images = len(id_list)
        count = 0
        while count < N_images:
            if (count >= N_images):
                count = 0

            out = np.zeros((N_classes), dtype=np.float32)
            out[id_list[count]] = 1

            count += 1
            yield out # one-hot vector of length N_classes

    def create_dataset_input_target(self):
        ds_input_target = tf.data.Dataset.zip((self.ds_input, self.ds_target)).shuffle(self.N_train).prefetch(tf.data.experimental.AUTOTUNE)
        return ds_input_target


    def build_model(self):

        x0 = tf.keras.layers.Input((self.resolution,self.resolution,3))

        # normalize approximately by shifting
        x = tf.keras.layers.Lambda(lambda x: x - 0.5)(x0)

        #x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        #x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        #x = tf.keras.layers.MaxPooling2D()(x)
        #x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        #x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        #x = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        #x = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        #x = tf.keras.layers.MaxPooling2D()(x)
        #x = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        #x = tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        #x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(700, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L1(0.01))(x)
        x = tf.keras.layers.Dense(self.N_classes, activation='softmax')(x)

        model = tf.keras.Model(x0,x)

        model.summary()

        model.compile(
            #optimizer=tf.keras.optimizers.Adam(),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )
        return model


# end Preprocessing

class Train:
    def __init__(self, N_classes, N_batch, resolution, all_img, all_label, all_img_test, file_list_test):
        self.N_classes = N_classes
        self.N_batch = N_batch
        self.resolution = resolution
        self.all_img = all_img
        self.all_label = all_label
        self.all_img_test = all_img_test
        self.file_list_test = file_list_test

        self.model = self.build_model()

        self.test_predictions = []

        return

    def build_model(self):
        # build model
        x0 = tf.keras.layers.Input((self.resolution, self.resolution, 3))

        # normalize approximately by shifting
        x = tf.keras.layers.Lambda(lambda x: x - 0.5)(x0)

        x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', trainable=True)(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(700, activation='relu', kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.L1(0.01))(x)
        x = tf.keras.layers.Dense(self.N_classes, activation='softmax')(x)

        model = tf.keras.Model(x0, x)
        model.summary()

        model.compile(
            # optimizer=tf.keras.optimizers.Adam(),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )
        return model

    def train(self): # train using all_img, all_label

        # tf.data from tensor slices
        ds_input_target = tf.data.Dataset.from_tensor_slices((self.all_img, self.all_label))

        # train
        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logs,
            histogram_freq=1
        )
        history = self.model.fit(
            ds_input_target,
            batch_size=self.N_batch,
            epochs=5,
            # steps_per_epoch=self.N_train,
            callbacks=[tboard_callback]
        )
        self.model.save('./models/model_2/')

        return

    def load_model(self):
        self.model = tf.keras.models.load_model('./models/model_2/')
        return

    def test(self):
        ds = tf.data.Dataset.from_tensor_slices((self.all_img_test))

        # predict and save csv
        result = self.model.predict(ds, batch_size=self.N_batch)
        self.test_predictions = result

        header = ['id', 'label']
        rows = []
        for obj in result:
            idx = np.argmax(obj)
            # convert integer to label string
            label = x.get_label(idx)
            rows.append(label)

        ids = []
        ids2 = []
        for obj in self.file_list_test:
            id = obj.split('\\')[-1]  # ###.jpg
            ids.append(id)
        for obj in ids:
            id = obj.split('.')[0]  # ###
            ids2.append(id)

        rows2 = []
        for obj in list(zip(ids2, rows)):
            rows2.append([obj[0], obj[1]])

        # write to csv
        file = 'test_2.csv'
        with open(file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(rows2)
        return
# end Train

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    x = Preprocessing()
    ds_input = x.ds_input
    ds_target = x.ds_target

    print('--------taking from ds_input-----------')
    N = 2
    for img in ds_input.take(N):
        print(img.shape)

        plt.figure()
        for i in range(x.N_batch):

            plt.subplot(1,x.N_batch,i+1)
            #plt.imshow(sum(img)/x.N_batch)
            plt.imshow(img[i,:])

    print('---------taking from ds_target----------')
    for label in ds_target.take(N):
        print(label)

    print('done preprocessing')

    # ---------------------------------------------------
    # make a new instance and train
    #print('beginning to train...')
    #x2 = Preprocessing(train=True)

    # ---------------------------------------------------
    # try another way to train
    x2 = Preprocessing(train=False)
    count = 0
    for (img,label) in x2.ds_input_target.take(-1): # take batches until end
        print(img.shape)
        print(label.shape)
        if count==0:
            all_img = img
            all_label = label
            count += 1
        else:
            all_img = np.concatenate([all_img, img])
            all_label = np.concatenate([all_label, label])
            count += 1
    print(all_img.shape)
    print(all_label.shape)

    count = 0
    for img in x2.ds_input_test.take(-1):
        if count==0:
            all_img_test = img
            count += 1
        else:
            all_img_test = np.concatenate([all_img_test, img])
            count += 1
    print(all_img_test.shape)

    # show example i
    i = 0
    plt.figure()
    plt.imshow(all_img[i])
    plt.title(x2.get_label(np.argmax(all_label[i])))

    # train
    xx = Train(x2.N_classes, x2.N_batch, x2.resolution, all_img, all_label, all_img_test, x2.file_list_test)
    xx.train()
    xx.test()

    print('-- done')

