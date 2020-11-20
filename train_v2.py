import tensorflow as tf
import pickle
from preprocessing import *

if __name__ == '__main__':

    with open('train_data_v2.pkl', 'rb') as f:
        N_classes, N_batch, resolution, all_img, all_label, all_img_test, file_list_test, unique_labels = pickle.load(f)

    # show example i
    i = 0
    plt.figure()
    plt.imshow(all_img[i])
    plt.title(unique_labels[np.argmax(all_label[i])])

    # train
    xx = Train(N_classes, N_batch, resolution, all_img, all_label, all_img_test, file_list_test, unique_labels)
    xx.train(epochs=80, lr=1e-3)
    xx.test()

    print('-- done')

    # \logs\20201119-030932
    # \logs\20201120-010001
    # overfitting, even with augmentation
    # L1 regularization @ second last Dense layer doesn't help