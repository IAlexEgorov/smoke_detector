from matplotlib import pyplot as plt                                                                            #  Подключаем библиотеки:
                                                                                                                #  
import os                                                                                                       #  системных функций
import json                                                                                                     #  работы с json 
                                                                                                                #  
import numpy as np                                                                                              #  общих мат. функций, операций и массивов 
import pandas as pd                                                                                             #  обработки и анализа данных
from tqdm import tqdm                                                                                           #  индикатор прогресса
from sklearn.model_selection import train_test_split                                                            #  общего машинного обучения

import cv2                                                                                                      #  
import cv2 as cv                                                                                                #  компьютерного зрения 
from pandas import date_range, Series, DataFrame, read_csv, qcut                                                #  обработки и анализа данных
import random                                                                                                   #  случайных чисел
import pickle                                                                                                   #  сериализации и де сложных Python-объектов
from numpy.random import randn                                                                                  #  рандома из нампая
                                                                                                                #  
import tensorflow as tf                                                                                         #  для решения задач построения и тренировки нейронной сети
                                                                                                                #
import tensorflow.keras                                                                                         #  нейронных сетей 
from tensorflow.keras import backend as K                                                                       #  и доп 
from tensorflow.keras.models import Model                                                                       #  функций 
                                                                                                                #  
#from keras.utils import multi_gpu_model                                                                         #  с фукцией параллелизма
                                                                                                                #  
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, Activation                          # | 
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, SeparableConv2D   # | функций работы со слоями 
from tensorflow.keras.layers import Convolution2DTranspose, concatenate, UpSampling2D                           # |
from tensorflow.keras.losses import binary_crossentropy                                                         #  функций потерь (кросэнтропия)
from tensorflow.keras.callbacks import Callback, ModelCheckpoint                                                #  функции о внутреннем состоянии модели во время обучения
                                                                                                                #  
                                                                                                                #  
print(tf.keras.__version__)                                                                                     #  Кераса
print(tf.__version__)                                                                                           #  Выводим данные по версии тензорфлоу
print(cv2.__version__)                                                                                          #  ОупенСиВи
                                                                                                                #  
def dice_coef(y_true, y_pred, smooth=1):                                                                        #  
    y_true_f = K.flatten(y_true)                                                                                #  
    y_pred_f = K.flatten(y_pred)                                                                                #  
    intersection = K.sum(y_true_f * y_pred_f)                                                                   #  
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)                          #  
                                                                                                           

def dice_loss(y_true, y_pred):                                                                                  # Сглаживает веса
    smooth = 1.                                                                                                 # коэф сглаживания 
    y_true_f = K.flatten(y_true)                                                                                # Переводит в 1д 
    y_pred_f = K.flatten(y_pred)                                                                                # Переводит в 1д 
    intersection = y_true_f * y_pred_f                                                                          # Пересечение
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)                                                            # Подсчитывает потери ???
    return 1. - score                                                                                    
                                                                                                                  
                                                                                                                  


def bce_dice_loss(y_true, y_pred):                                                                              #  Измененная функция потерь
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)                                      #  ???кросэнтропия+корректировка точности???



def load_grayscale(img_path):                                                                                   #  | Загружает картинку в ЧБ
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                                                            #  |
    return img                                                                                                  #  Возвращает ее
                                                                                                                  
                                                                                                                 

                                                                                                                
def load_rgb(img_path):                                                                                         #  Загружает картинку с переводом в BGR
    img = cv2.imread(img_path)                                                                                  #  Считываем картинку
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                                  #  Переводим картинку в BGR
    return img                                                                                                  #  Возвращает ее
                                                                                                                  
                                                                                                                  



def image_show(image, mask, fontsize = 12):                                                                     #  
    f, ax = plt.subplots(2, 1, figsize=(12, 12))                                                                #  Создает фигуру и оси 
    ax[0].imshow(image, interpolation='bilinear', cmap='gray')                                                  #  показывает картинку на графике
    ax[0].set_title('Image', fontsize = fontsize)                                                               #  дает ей название
    ax[1].imshow(mask, interpolation='bilinear', cmap='gray')                                                   #  показывает маску картинки 
    ax[1].set_title('Result', fontsize = fontsize)                                                              #  дает название: "результат" 
    plt.show()                                                                                                  #  Выводит график на экран 
                                                                                                                


def infer_image(model, img):                                                                                    #  
                                                                                                                #  
    x = img.astype('float32') / 255.                                                                            #  Массив из цветов деленных на 255 
    x = np.expand_dims(x, axis = 0)                                                                             #  Добавляет ось слева \ пример: (1, 0.43)
    pred = model.predict(x)                                                                                     #  Предположение модели, которое на выходе даст 
                                                                                                                #  массив из предположений на пиксели 
    return pred[0,:,:,0]                                                                                        #  Возвращает предсказанную модель
                                                                                                                #  
                                                                                                                #  



def filter_model(input_shape, kernel_size, n_filters = 3):                                                      #  Создает готовую модель сети 
                                                                                                                #  
    input_img = Input(shape = input_shape)                                                                      #  Создание тензора (типа начала нейронки - потом слои идут)
    kernel_tuple = (kernel_size, kernel_size)                                                                   #  Кортеж ядра 
                                                                                                                #  
    x = Conv2D(n_filters, kernel_tuple, activation='relu', padding='same')(input_img)                           #  Создание сверточного 2д слоя после входящего слоя
    output_img = Conv2D(1, kernel_tuple, activation='sigmoid', padding='same')(x)                               #  Создание сверточного 2д слоя на выход после скрытого 
                                                                                                                #  
    model = Model(input_img, output_img)                                                                        #  Создание модели сети  
                                                                                                                #  
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])                                    #  Компиляция модели сети 
                                                                                                                #  
    model.summary()                                                                                             #  Печать сводки сети 
    return model                                                                                                #  
                                                                                                                #  
                                                                                                                #  


def test_model(modelname = 'default_filter', csvname = 'smoke_test.csv', grayscale = False):                    #  
                                                                                                                #  
    input_size = (720, 1280)                                                                                    #  Задаем размер картинок
    kernel_size = 11                                                                                            #  Размер ядра
    n_filters = 1                                                                                               #  количество фильтров
    

    input_shape = (720, 1280, 3)
    kernel_size = 11

    smokefilter = filter_model(input_shape, kernel_size, 3)


    ### Load the weights                                                                                      
    h5name = modelname + '-weights.h5'                                                                        
    smokefilter.load_weights(h5name)                                                                                   
                                                                                                              
    data = pd.read_csv(csvname)                                                                               
    n_samples = len(data)                                                                                     

    cap = cv2.VideoCapture("dim.mp4")
    
    #while(cap.isOpened()):
    #    ret, frame = cap.read()
    #    if ret == True:
    #        msk = infer_image(smokefilter, frame)                                                             
    #        cv2.imshow('Frame', frame)
    #        if cv2.waitKey(25) & 0xFF == ord('q'):
    #            break
    #    else:
    #
    #      break   

    fig, ax = plt.subplots()
    co=0
    while(cap.isOpened()):
        if(co<24):
            co+=1
            ret, frame = cap.read()
            continue
        else:
            co=0
            ret, frame = cap.read()
            if ret == True:
                msk = infer_image(smokefilter, frame) 
                msk=msk*100;
                cv2.imshow('Frame', msk)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
              break

                                                                                                                #  
if __name__ == '__main__':                                                                                      #  
                                                                                                                #  
    #test_model(modelname = 'smokefilter_3conv3_11x11x1')          
                                                  
    test_model(modelname = 'smokefilter_dice', grayscale = False)    