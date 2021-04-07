#!/usr/bin/env python


import os
import numpy as np
import json
import torch
import torch.utils.data as Data
import random

from torch.autograd import Variable


def set_seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def load_data(training_data_location):
    result = np.loadtxt(open(training_data_location, "rb"), delimiter=",")
    x = result[:, 1]
    xt = torch.Tensor(x.reshape(-1, 1))
    
    y = result[:, 0]
    yt = torch.Tensor(y.reshape(-1, 1))
    
    return (xt, yt)


def prepare_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.Dropout(0.01),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.Dropout(0.01),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.Dropout(0.01),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.Dropout(0.01),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.Dropout(0.01),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.Dropout(0.01),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.Dropout(0.01),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1),
    )
        
    return model



def prepare_data_loader(x, y, batch_size):
    dataset = Data.TensorDataset(x, y)

    data_loader = Data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False, num_workers=2)
    
    return data_loader



def train(model, x, y, epochs=200, learning_rate = 0.001, batch_size=100):
    data_loader = prepare_data_loader(x=x, y=y, batch_size=batch_size)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        for step, (batch_x, batch_y) in enumerate(data_loader): 
            prediction = model(Variable(batch_x))     

            loss = loss_fn(prediction, Variable(batch_y))     

            optimizer.zero_grad()   
            loss.backward()         
            optimizer.step()        

        if (e % 10 == 0):
            print("Iteration:", e, "\t| Loss:", loss.item())
            
    
    return model


def main(model_dir, train_path, epochs=200, 
         learning_rate=0.001, batch_size=100):                                  
    
    set_seed()
    model = prepare_model()
    
    x, y = load_data(train_path)
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    
    model = train(model=model, 
                  x=x, 
                  y=y, 
                  epochs=epochs, 
                  learning_rate=learning_rate,
                  batch_size=batch_size)
    
    print(model)
        
    torch.save(model.state_dict(), 
               os.path.join(model_dir, "model.pth"))
                                      
                                      
if __name__ == "__main__":
    model_dir = "/opt/ml/model"
    train_path = "/opt/ml/input/data/train/training_data.csv"
                                      
    main(model_dir=model_dir,
         train_path=train_path,
         epochs=2000,
         learning_rate=0.001,
         batch_size=100)
