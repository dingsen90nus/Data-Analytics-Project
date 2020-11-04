#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:33:24 2020

@author: dingsen
"""
    
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
import tensorflow as tf


#### Step 1: Data Preparation### 

train = pd.read_excel("data_train.xlsx").dropna(axis=0)
test = pd.read_excel("data_test.xlsx").dropna(axis=0)

def prepare_data(df_data):
    selected_cols = ['Product Sub Center',
                  'Order Product Quantity',
                  'Delivery Priority Code','Product Shipping Point',
                  'Product Supplier','Order Type' , 'Product Type', 
                  'Delivery Time (days)']
    df= df_data[selected_cols]
    df['Product Sub Center'] =df['Product Sub Center'].map({'ABC1':0, 'ABC2':1}).astype(int)
    df['Product Supplier'] =df['Product Supplier'].map({'S1':0, 'S2':1, 'S4':2}).astype(int)
    df['Order Type'] =df['Order Type'].map({'T1':0, 'T2':1}).astype(int)
    df['Product Type'] =df['Product Type'].map({'PT1':0, 'PT2':1}).astype(int)
    df['Product Shipping Point'] =df['Product Shipping Point'].map({'J1':0, 'J10':1, 'J2':2, 'J8':3}).astype(int)
    
    ndarray_data= df.values
    df =df.sample(frac=1)

    features=ndarray_data[:,:-1]
    label = ndarray_data[:,-1]
    
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    norm_features = minmax_scale.fit_transform(features)
    return norm_features,label


x_test, y_test = prepare_data(test)
x_train, y_train = prepare_data(train)


##### Step 2: Model Building/Fitting#####

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units = 64, 
                                input_dim = 7, 
                                use_bias = True,
                                kernel_initializer ='uniform',
                                bias_initializer = 'zeros',
                                activation ='relu'))

model.add(tf.keras.layers.Dense(units= 32,
                                activation = 'relu'))


model.add(tf.keras.layers.Dense(units= 1,
                                activation = 'relu'))


model.summary()


model.compile(optimizer = tf.keras.optimizers.Adam(0.003),
              loss='mse',
              metrics = ['accuracy'])

train_history=model.fit(x = x_train,
                        y = y_train,
                        validation_split = 0.2,
                        epochs = 100, 
                        batch_size = 40,
                        verbose = 2)

##### Step 3: Accuracy/Loss Curve####

train_history.history.keys()

def visu_train_history(train_history, train_metric, validation_metric):
        plt.plot(train_history.history[train_metric])
        plt.plot(train_history.history[validation_metric])
        plt.title('Train History')
        plt.ylabel(train_metric)
        plt.xlabel('epoch')
        plt.legend(['train','validation'],loc='upper left')
        plt.show()
        
        
visu_train_history(train_history, 'accuracy','val_accuracy')
visu_train_history(train_history, 'loss','val_loss')


#### Step 4: Model Evaluation####
evaluate_result = model.evaluate(x = x_test, y = y_test)

evaluate_result
model.metrics_names


#### Model Evaluation2####
x_features, y_label = prepare_data(test)
predicted_days = model.predict(x_features)
test.insert(len(test.columns),'predicted days', predicted_days)
