from matplotlib import pyplot as plt
import plaidml.keras
plaidml.keras.install_backend()
import keras

import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
from pandas import date_range, Series,DataFrame, read_csv, qcut
import random
import pickle
import os
from numpy.random import randn

from keras.models import load_model

from tensorflow.keras.optimizers import RMSprop,Adam

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.layers import Convolution2DTranspose, concatenate, UpSampling2D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

print(tf.__version__)
print(tf.keras.__version__)
print(cv2.__version__)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def load_grayscale(img_path):
    ### for GRAYSCALE # img = np.expand_dims(img, axis=-1)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img



def load_data(csv_file):
    data = pd.read_csv(csv_file)
    n_samples = len(data)
    lx = []
    ly = []
    for idx in range(n_samples):
        print(idx, ': ', data['IMAGE'].iloc[idx], ', mask: ', data['MASK'].iloc[idx])

        img = cv2.resize(load_rgb(data['IMAGE'].iloc[idx]), (640, 360))
        lx.append(img.astype('float32') / 255.)

        print(img.shape)
        msk = cv2.resize(load_grayscale(data['MASK'].iloc[idx]), (640, 360))
        msk = np.expand_dims(msk, axis=-1)
        ly.append(msk.astype('float32') / 255.)

    size_i = lx[0].shape
    xx = np.array(lx)
    xx = np.reshape(xx, (len(data), *size_i))

    size_m = ly[0].shape
    yy = np.array(ly)
    yy = np.reshape(yy, (len(data), *size_m))

    return xx, yy

def load_rgb(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_show(img):
    fig, ax = plt.subplots()
    im = ax.imshow(img, interpolation='bilinear', cmap='gray')
    plt.show()

def test_rgbimage(model, img_path):

    x = load_rgb(img_path)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis = 0)
    pred = model.predict(x)
    
    image_show(pred[0,:,:,0])


def infer_image(model, img):        
                                    
    x = img.astype('float32') / 255.
    x = np.expand_dims(x, axis = 0) 
    pred = model.predict(x)         
                                    
    return pred[0,:,:,0]    

def return_graf(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    plt.title('Training and validation loss')

    plt.legend(loc=0)
    plt.figure()
    plt.show()

def check_rez(model, path="11.jpg"):
    frame = load_rgb(path)
    frame = cv2.resize(frame, (640, 360))
    p_img = infer_image(model, frame)
    image_show(frame)
    image_show(p_img)

def filter_model(input_shape, kernel_size, n_filters = 3):

    kernel_tuple = (kernel_size, kernel_size)
    
    input_img = Input(shape = input_shape)
    x = Conv2D(n_filters, kernel_tuple, activation='relu', padding='same')(input_img)
    x = Dropout(0.25)(x)
    output_img = Conv2D(1, kernel_tuple, activation='sigmoid', padding='same')(x)

    model = Model(input_img, output_img)

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.summary()
    return model
        

def train_filter(fit_bool, modelname = 'smokefilter_2conv3_11x11x3'):

    #input_shape = (720, 1280, 3)
    input_shape = (360, 640, 3)
    kernel_size = 11

    smokefilter = filter_model(input_shape, kernel_size, 3)

    h5name = modelname + '-weights.h5'                                                                         
    smokefilter.load_weights(h5name)

    if(fit_bool==True):
        
        x_train, y_train = load_data('smoke_train.csv')
        x_valid, y_valid = load_data('smoke_valid.csv')

        history = smokefilter.fit(x_train, y_train,
                        epochs = 40,
                        batch_size = 2,
                        shuffle = True,
                        validation_data = (x_train, y_train)
                       )

        h5name = modelname + '-weights.h5'
        smokefilter.save_weights(h5name)
    
        print(history)
        #return_graf(history)

        with open(modelname + '-architecture.json', 'w') as f:
            f.write(smokefilter.to_json())
    else:
        check_rez(smokefilter, "images/s4/00008_000.jpg")
        #check_rez(smokefilter)


        #cap = cv2.VideoCapture("./videos/1.mp4"); 
        #
        #while(cap.isOpened()):
        #    ret, frame = cap.read()
        #    frame = cv2.resize(frame, (640, 360))
        #    if ret == True:
        #      p_img = infer_image(smokefilter, frame)*100
        #      cv2.imshow('Frame',p_img)
        #      if cv2.waitKey(25) & 0xFF == ord('q'):
        #        break
        #    else:
        #      break


if __name__ == '__main__':
    train_filter(False, 'smokefilter_dice')