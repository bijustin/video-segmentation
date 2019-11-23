#!/usr/bin/env python3

import glob
import os
import numpy as np
import keras
from keras.preprocessing import image as kImage
import sys
import cv2

from FgSegNet_module import VGG16


dataset_paths = {
    'CDnet2014': 'dataset/dataset2014/dataset',
}

def getData():
    """
    Load training dataset
    """
    X = []
    Y = []
    for _, dataset_path in dataset_paths.items():
        if not os.path.isdir(dataset_path):
            # download and unzip
            pass
        X_list = sorted(glob.glob(os.path.join(dataset_path, '*', '*', 'input', '*.jpg')))
        y_list = sorted(glob.glob(os.path.join(dataset_path, '*', '*', 'groundtruth', '*.png')))

        for i in range(len(X_list)):
            # load trainning images
            x = kImage.load_img(X_list[i])
            x = kImage.img_to_array(x)
            X.append(x)

            # load ground truth label and encode it to label 0/1 (black and white as the final mask)
            y = kImage.load_img(y_list[i], grayscale=True)
            y = kImage.img_to_array(y)
            y /= 255.0
            y = np.floor(y)
            Y.append(y)
        X = np.asarray(X)
        Y = np.asarray(Y)

        # shuffle the training dataset
        idx = list(range(X.shape[0]))
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]
        return X, Y


def train(X, Y, id):
    """ 
    Train models
    """
    # initialize the model
    model =VGG16()
    
    # stop the training early when validation loss stops improving in 10 epochs
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)

    # reduce the learning rate by a factor of 10 when validation loss stops improving in 5 epochs
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    model.fit(X, Y, batch_size=1, epochs=100, verbose=2, validation_split=0.2, callbacks=[reduce, early], shuffle=True)
    model.save('trained_model/model_{}.h5'.format(id))

    

if __name__ == "__main__":
        X_train, Y_train = getData()
        train(X_train, Y_train, 0)