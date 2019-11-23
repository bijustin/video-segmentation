#!/usr/bin/env python3

import keras
from keras.models import Model
from keras.layers import Dropout, Deconvolution2D, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


def VGG16():
        """ 
        default VGG16 model provided by keras library
        """
        net_input = Input(shape =(None, None, 3))
        vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=net_input)
        for layer in vgg16.layers[:17]:
            layer.trainable = False
        
        x = vgg16.layers[-2].output # 2nd layer from the last, block5_conv3
        
        ### Decoder
        x = Deconvolution2D(256, (3,3), strides=(2,2), activation='relu', padding='same')(x)
        x = Deconvolution2D(128, (3,3), strides=(2,2), activation='relu', padding='same')(x)
        x = Deconvolution2D(64, (3,3), strides=(2,2), activation='relu', padding='same')(x)
        x = Deconvolution2D(32, (3,3), strides=(2,2), activation='relu', padding='same')(x)
        x = Deconvolution2D(1, (1,1), activation='sigmoid', padding='same')(x)
        
        model = Model(inputs=vgg16.input, outputs=x)
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.RMSprop(lr=5e-4), metrics=['accuracy'])
        return model


# class VGG_Module(object):

#     def __init__(self):
#         pass
# 
    # def VGG16(self, x):
    #     """ 
    #     Convolutional Network for classification and Detection
        
    #     x: (batch, steps, channels)
        
    #     Structure:
    #         Input -> Conv 1-1 -> Conv 1-2 -> Pooling -> Conv 2-1 -> Conv 2-2
    #             -> Pooling -> Conv 3-1 -> Conv 3-2 -> Conv 3-3 -> Pooling
    #             -> Conv 4-1 -> Conv 4-2 -> Conv 4-3 -> Pooling -> Conv 5-1 
    #             -> Conv 5-2 -> Conv 5-3 -> Pooling -> Dense -> Dense -> Dense
    #     """
    #     # Block 1
    #     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
    #     x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    #     a = x
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    #     # Block 2
    #     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    #     x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    #     b = x
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    #     # Block 3
    #     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    #     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    #     x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    
    #     # Block 4
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    #     # prevent overfitting
    #     x = Dropout(0.5, name='dr1')(x)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    #     x = Dropout(0.5, name='dr2')(x)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    #     x = Dropout(0.5, name='dr3')(x)
        
    #     return x, a, b