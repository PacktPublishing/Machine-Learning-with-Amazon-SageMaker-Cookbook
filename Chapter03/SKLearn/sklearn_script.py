#!/usr/bin/env python

import os
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib


def set_seed():
    np.random.seed(0) 


def model_fn(model_dir):
    path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(path)
    
    return model


def load_data(training_data_location):
    result = np.loadtxt(open(training_data_location, "rb"), delimiter=",")
    y = result[:, 0]
    x = result[:, 1]
    
    return (x, y)


def prepare_model(epochs=1000):
    model = MLPRegressor(hidden_layer_sizes=(10,10,10,10,10), 
                         activation='relu', 
                         solver='adam', 
                         max_iter=2000, 
                         verbose=True,
                         batch_size=100,
                         learning_rate='adaptive',
                         n_iter_no_change=2000,
                         early_stopping=True,
                         tol=0.01,
                         random_state=0)  
    
    return model


def train(model, x, y):
    model.fit(x.reshape(-1, 1),y.reshape(-1, 1))
    
    return model


def main(model_dir, train_path, epochs=2000):
    set_seed()
    model = prepare_model(epochs=epochs)
    
    x, y = load_data(train_path)
    print(x.shape)
    print(y.shape)
    
    model = train(model, x, y)
    print(model)
            
    path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, path)    


if __name__ == "__main__":
    model_dir = "/opt/ml/model"    
    train_path = "/opt/ml/input/data/train/training_data.csv"
    
    main(model_dir=model_dir,
         train_path=train_path,
         epochs=2000)
