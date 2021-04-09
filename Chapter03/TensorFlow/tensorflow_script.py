#!/usr/bin/env python

import tensorflow as tf
import os
import numpy as np

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

from numpy.random import seed



def set_seed():
    seed(42)
    tf.random.set_seed(42)


def load_data(training_data_location):
    result = np.loadtxt(open(training_data_location, "rb"), delimiter=",")
    
    y = result[:, 0]
    x = result[:, 1]
    
    return (x, y)


def prepare_model():
    model = Sequential([
        Dense(100, activation=tf.nn.leaky_relu, 
                   input_shape=[1]),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(1)
      ])
    
    model.compile(loss='mean_squared_error', 
                  optimizer='adam')
    
    return model


def main(model_dir, train_path, val_path, 
         batch_size=200, epochs=2000):
    set_seed()
    
    model = prepare_model()
    model.summary()
        
    x, y = load_data(train_path)
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    
    x_val, y_val = load_data(val_path)
    print("x_val.shape:", x_val.shape)
    print("y_val.shape:", y_val.shape)

    model.fit(x=x, 
              y=y, 
              batch_size=batch_size, 
              epochs=epochs, 
              validation_data=(x_val, y_val))    
        
    tf.saved_model.save(
        model, 
        os.path.join(model_dir, '000000001'))


if __name__ == "__main__":
    data_path = "/opt/ml/input/data"
    model_dir = "/opt/ml/model"
    train_path = f"{data_path}/train/training_data.csv"
    val_path = f"{data_path}/validation/validation_data.csv"
    
    main(model_dir=model_dir,
         train_path=train_path,
         val_path=val_path,
         batch_size=200,
         epochs=1000)