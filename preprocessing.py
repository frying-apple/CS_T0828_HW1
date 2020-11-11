import pathlib
import tensorflow as tf
from csv import reader
import numpy as np
import os
import operator

class Preprocessing:
    def __init__(self):
        self.file = pathlib.Path('.\\cs-t0828-2020-hw1\\')  # windows
        # self.filepath = pathlib.Path('') # ubuntu

        self.N_batch = 8

        self.resolution = 200 # resize/crop to square image of this size


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
        self.file_list_train = train_contents
        self.file_list_test = test_contents
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
            output_shapes=([tf.TensorShape([100,100,3])])
        )
        ds_input = ds_input.batch(self.N_batch)
        return ds_input

    @staticmethod
    def generator_input(resolution, file_list): # TODO
        N_images = len(file_list)
        count = 0
        while True:
            if (count >= N_images):
                count = 0

            # resize and crop (center) image to (resolution,resolution,3)

            # get image
            img = tf.io.read_file(file_list[count])
            img = tf.image.decode_png(img)
            img = tf.image.convert_image_dtype(img,tf.float32)
            img = img.numpy()

            # get dimensions
            img_shape = img.shape

            # resize to minimum dimension
            min_res = 150 # px
            if img_shape[0] < min_res:
                resize_factor = min_res/img_shape[0]
            elif img_shape[1] < min_res:
                resize_factor = min_res/img_shape[1]
            else:
                resize_factor = 1.0

            # resize to at least min_res; keep aspect ratio
            img = tf.image.resize(img, (img_shape[0]*resize_factor, img_shape[1]*resize_factor), antialias=True)

            # random crop a square
            output_size = 100 # px
            img = tf.image.random_crop(img, [output_size,output_size,3])  # should be smaller than min_res

            # optionally, also change image brightness, saturation, hue, etc. using dataset map
            # see https://www.tensorflow.org/tutorials/images/data_augmentation

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
        while True:
            if (count >= N_images):
                count = 0

            out = np.zeros((N_classes), dtype=np.float32)
            out[id_list[count]] = 1
            yield out # one-hot vector of length N_classes

    def save_npz(self):
        '''
        save numpy arrays to disk so can load for training/testing
        '''
        return 0


# end Preprocessing

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
    #x.save_npz()

    print('done preprocessing')
