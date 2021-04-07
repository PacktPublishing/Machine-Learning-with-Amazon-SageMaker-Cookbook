#!/usr/bin/env python

import os
import torch
import numpy as np


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


def model_fn(model_dir):
    model = prepare_model()
    
    path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model


if __name__ == "__main__":
    model_dir = "/opt/ml/model"
 
    model = model_fn(model_dir)
    
    print('-----------')
    print('model([100]):', model(torch.Tensor([100])))
    print('model([200]):', model(torch.Tensor([200])))