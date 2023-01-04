# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:49:23 2022

@author: danie
"""

import torch.nn as nn
import torch.nn.functional as F
import os
os.chdir(r"C:\Users\danie\UiB\Master\Data_gen")


class MLP(nn.Module):
    
    def __init__(self, n_species, hidden_layer_size, n_hidden_layer = 2):
        super(MLP,self).__init__()
        self.input = nn.Linear(n_species, hidden_layer_size)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_layer_size, hidden_layer_size) 
                                            for i in range(1,n_hidden_layer)])
        self.output = nn.Linear(hidden_layer_size, 1)
    
    def forward(self, x):
        x = F.relu(self.input(x))
        for layer in self.linear_layers:
            x = F.relu(layer(x))
        x = self.output(x)

        return x

if __name__ == "__main__":
    net = MLP(8,8,2)
    for name, param in net.named_parameters():
        print(f'Name = {name}\nParam = {param.shape}')
        