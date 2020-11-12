import tensorflow as tf
from preprocessing import Preprocessing
import numpy as np
import csv

if __name__ == '__main__':
    x = Preprocessing()

    # load model
    x.model = tf.keras.models.load_model('./models/model/')

    # test
    print('testing...')
    print(x.N_test)

    # reminder: tf dataset is infinite loop generator
    N_steps = np.ceil(x.N_test / x.N_batch).astype(int)
    y_pred = x.model.predict(x.ds_input_test, steps=N_steps) # steps = ceil(N_test / x.N_batch); take first N_test items
    print(y_pred.shape)

    # take first N_test items
    y_pred_2 = y_pred[0:x.N_test,:]
    print(y_pred_2.shape)

    header = ['id','label']
    rows = []
    for obj in y_pred_2:
        #print(obj.shape)

        idx = np.argmax(obj)

        # convert integer to label string
        label = x.get_label(idx)
        #print(label)
        #rows.append([str(idx),label]) # want id from filename
        rows.append(label)

    ids = []
    ids2 = []
    for obj in x.file_list_test:
        #print(obj)
        id = obj.split('\\')[-1] # ###.jpg
        ids.append(id)
    for obj in ids:
        id = obj.split('.')[0] # ###
        ids2.append(id)

    rows2 = []
    for obj in list(zip(ids2,rows)):
        rows2.append([obj[0],obj[1]])
    #print(rows2)

    # write to csv
    file = 'test.csv'
    with open(file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(rows2)

    print('-- done')