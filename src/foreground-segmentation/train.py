#!/usr/bin/env python3

import glob
import os
import numpy as np
import keras
from keras.preprocessing import image as kImage
import sys
import cv2
from math import floor

from FgSegNet_module import VGG16

BATCH_SIZE = 16


dataset_paths = {
    'CDnet2014': 'dataset/dataset2014/dataset',
}

def getDataList():
    """
    Load training dataset
    """
    X_list = []
    y_list = []
    for _, dataset_path in dataset_paths.items():
        if not os.path.isdir(dataset_path):
            # download and unzip
            pass
        X_list.append(sorted(glob.glob(os.path.join(dataset_path, '*', '*', 'input', '*.jpg'))))
        y_list.append(sorted(glob.glob(os.path.join(dataset_path, '*', '*', 'groundtruth', '*.png'))))
    X_list = np.array(X_list)
    y_list = np.array(y_list)
    idx = list(range(X_list.shape[0]))
    np.random.shuffle(idx)
    X_list = X_list[idx]
    Y_list = y_list[idx]

    return X_list, y_list


def train(X, Y, model):
    """ 
    Train models
    """
    
    # stop the training early when validation loss stops improving in 10 epochs
    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)

    # reduce the learning rate by a factor of 10 when validation loss stops improving in 5 epochs
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

    model.fit(X, Y, batch_size=1, epochs=1, verbose=2, validation_split=0.2, callbacks=[reduce, early], shuffle=True)

    

if __name__ == "__main__":
        X_list, Y_list = getDataList()
        X_list = np.squeeze(X_list, axis=0)
        Y_list = np.squeeze(Y_list, axis=0)
        # ignore the last few images
        batch_num = floor(X_list.shape[0]/BATCH_SIZE)

        # initialize the model
        model =VGG16()

        for i in range(batch_num):
            X = []
            Y = []
            # one batch of dataset
            # paths of the images in current batch
            X_current = X_list[i: i + BATCH_SIZE]
            Y_current = Y_list[i: i + BATCH_SIZE]
            
            for x_path, y_path in zip(list(X_current), list(Y_current)):
                # load images for current batch
                x = kImage.load_img(x_path)
                x = kImage.img_to_array(x)
                X.append(x)

                # load ground truth label and encode it to label 0/1 (black and white as the final mask)
                y = kImage.load_img(y_path, grayscale=True)
                y = kImage.img_to_array(y)
                y /= 255.0
                y = np.floor(y)
                Y.append(y)

            X = np.asarray(X)
            Y = np.asarray(Y) 

            # train the model based on current batch
            train(X, Y, model)
            
        model.save('trained_model/model.h5')

