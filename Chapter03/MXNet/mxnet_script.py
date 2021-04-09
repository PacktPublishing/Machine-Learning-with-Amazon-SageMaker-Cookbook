#!/usr/bin/env python

import os
import numpy as np
import mxnet as mx

from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.contrib.estimator import Estimator
from mxnet.random import seed


def set_seed():
    seed(0)
    np.random.seed(0)


def model_fn(model_dir):
    model = prepare_model()
    path = os.path.join(model_dir, "model.params")
    model.load_parameters(path)
    
    return model


def load_data(training_data_location):
    file_object = open(training_data_location, "rb")
    result = np.loadtxt(file_object, delimiter=",")
    x = result[:, 1]
    y = result[:, 0]
    
    xt = mx.nd.array(x.reshape(-1, 1))
    yt = mx.nd.array(y.reshape(-1, 1))
    
    return (xt, yt)


def prepare_model():
    model = nn.HybridSequential()

    with model.name_scope():
        model.add(nn.Dense(100, activation="relu"))
        model.add(nn.Dense(100, activation="relu"))
        model.add(nn.Dense(1))
        
    model.initialize()
        
    return model


def prepare_data_loader(x, y, batch_size):
    dataset = mx.gluon.data.dataset.ArrayDataset(x, y)
    data_loader = mx.gluon.data.DataLoader(
        dataset, 
        batch_size=batch_size)
    
    return data_loader


def train(model, x, y, x_val, y_val, 
          batch_size=100, epochs=1000, 
          learning_rate=0.002):
    
    loss_fn = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(
        model.collect_params(), 
        'adam', 
        {'learning_rate': learning_rate})
        
    train_data_loader = prepare_data_loader(
        x, y, 
        batch_size=batch_size)
    
    val_data_loader = prepare_data_loader(
        x_val, y_val, 
        batch_size=batch_size)

    estimator = Estimator(
        net=model,
        loss=loss_fn,
        trainer=trainer)

    estimator.fit(
        train_data=train_data_loader,
        val_data=val_data_loader,
        epochs=epochs)
    
    return model


def main(model_dir, train_path, val_path,
         epochs=1000, batch_size=100, 
         learning_rate=0.002):
    
    set_seed()
    model = prepare_model()
        
    x, y = load_data(train_path)
    x_val, y_val = load_data(val_path)
    
    print("x.shape:", x.shape)
    print("y.shape", y.shape)
    print("x_val.shape:", x_val.shape)
    print("y_val.shape", y_val.shape)
    
    model = train(model, x, y, x_val, y_val,
                  batch_size=batch_size, 
                  epochs=epochs,
                  learning_rate=learning_rate)
    print(model)
    
    path = os.path.join(model_dir, "model.params")
    model.save_parameters(path)


if __name__ == "__main__":
    model_dir = "/opt/ml/model"
    data_path = "/opt/ml/input/data"
    train_path = f"{data_path}/train/training_data.csv"
    val_path = f"{data_path}/validation/validation_data.csv"
    
    main(model_dir=model_dir,
         train_path=train_path,
         val_path=val_path,
         epochs=1000,
         batch_size=100,
         learning_rate=0.002)
