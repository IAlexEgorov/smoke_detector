from matplotlib import pyplot as plt

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


def load_grayscale(img_path):                                                              
    ### for GRAYSCALE # img = np.expand_dims(img, axis=-1)                                 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                                       
    return img                                                                             
                                                                                           
                                                                                           
def load_rgb(img_path):                                                                    
                                                                                           
    img = cv2.imread(img_path)                                                             
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                             
    return img                                                                             
                                                                                           
                                                                                           
def load_data(csv_file):                                                                   
                                                                                           
    data = pd.read_csv(csv_file)                                                           
    n_samples = len(data)                                                                  
    lx = []                                                                                
    ly = []                                                                                
    for idx in range(n_samples):                                                           
        print(idx, ': ', data['IMAGE'].iloc[idx], ', mask: ', data['MASK'].iloc[idx])      
                                                                                           
        img = load_rgb(data['IMAGE'].iloc[idx])                                            
        ### for GRAYSCALE # img = np.expand_dims(img, axis=-1)                             
        lx.append(img.astype('float32') / 255.)                                            
                                                                                           
        print(img.shape)                                                                   
        msk = load_grayscale(data['MASK'].iloc[idx])                                       
        msk = np.expand_dims(msk, axis=-1)                                                 
        ly.append(msk.astype('float32') / 255.)                                            
                                                                                           
    size_i = lx[0].shape                                                                   
    ### adapt this if using `channels_first` image data format, need channel last          
    xx = np.array(lx)                                                                      
    xx = np.reshape(xx, (len(data), *size_i))                                              
                                                                                           
    size_m = ly[0].shape                                                                   
    yy = np.array(ly)                                                                      
    yy = np.reshape(yy, (len(data), *size_m))                                              
                                                                                           
    return xx, yy                                                                          
                                                                                           
                                                                                           
def filter_model(input_shape, kernel_size, n_filters = 3):                                 
                                                                                           
    input_img = Input(shape = input_shape)                                                 
    kernel_tuple = (kernel_size, kernel_size)                                              
                                                                                           
    x = Conv2D(n_filters, kernel_tuple, activation='relu', padding='same')(input_img)      
    output_img = Conv2D(1, kernel_tuple, activation='sigmoid', padding='same')(x)          
                                                                                           
    model = Model(input_img, output_img)                                                   
                                                                                           
    model.compile(loss = 'binary_crossentropy',    
                  optimizer = 'adam',              
                  metrics = ['accuracy'])          
                                                                                           
    model.summary()                                                                        
    return model                                                                           
                                                                                           
                                                                                           
def train_filter(modelname = 'default'):                                                   
                                                                                           
    x_train, y_train = load_data('smoke_train.csv')                                        
    print(x_train.shape, y_train.shape)                                                    
                                                                                           
    x_valid, y_valid = load_data('smoke_valid.csv')                                        
                                                                                           
    input_shape = (720, 1280, 3)                                                           
    kernel_size = 11                                                                       
                                                                                           
    smokefilter = filter_model(input_shape, kernel_size, 3)                                
    smokefilter.fit(x_train, y_train,                                                      
                    epochs = 20,                                                           
                    batch_size = 2,                                                        
                    shuffle = True,                                                        
                    validation_data = (x_train, y_train)                                   
                   )                                                                       
    h5name = modelname + '-weights.h5'                                                     
    smokefilter.save_weights(h5name)                                                       
    with open(modelname + '-architecture.json', 'w') as f:                                 
        f.write(smokefilter.to_json())                                                     
                                                                                            
                                                                                            
                                                                                            
if __name__ == '__main__':                                                                  
                                                                                            
    train_filter('smokefilter_accu')                                                        