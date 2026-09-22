#!/usr/bin/env python

from copy import deepcopy
import torch
import torch.optim as optim
from os import path

from rock_model import RockNetModel

from utils.dataset import create_dataloaders
from utils.network import run_test, run_training


def train_from_dataset(data, epochs=5):
    print(f"Training for dataset {data}")
    train_dataloader, test_dataloader = create_dataloaders(data)

    net = RockNetModel()
    optimizer = optim.Adam(net.parameters(), lr=0.0025)

    run_training(epochs, train_dataloader, net, optimizer)
    acc = run_test(test_dataloader, net)
    
    print(f'Accuracy: {acc}')
    
    data_file_name, _ = path.splitext(data)
    model_file_name = f"{data_file_name}-nn-model.pt"
    torch.save(deepcopy(net.state_dict()), model_file_name)
    
    print("Done!")
    
        
    