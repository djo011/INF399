# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:15:44 2023

@author: danie
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\danie\repos\INF399\utils")
from NeuralNetwork import MLP
from Datasets import GeneExpressionData
os.chdir(r"C:\Users\danie\repos\INF399\dataGeneration")
from simdata_generation import CrausteModel, CellApoModel

def test_models(models, init_val, n_species):
    with torch.no_grad():
        for model in models: model.eval()
        
        X = torch.tensor(init_val)
        predictions = torch.zeros(60, n_species)
        predictions[0] = X
        for n in range(1,60):
            x_n_pred = torch.zeros((n_species,))
            for k in range(0,n_species):
                out = models[k](X.float())
                x_n_pred[k] = out
            predictions[n] = x_n_pred
            X = x_n_pred
        
        return predictions
    
def plot_predictions(predictions):
    data_gen = CrausteModel()
    base_data = data_gen.base_data
    time_t = data_gen.time_t
    t_scale = data_gen.t_scale
    t = [i for i in range(0,60)]
    fig, axs = plt.subplots(math.ceil(predictions.shape[1]/2),2, figsize = (15,15))
    fig.suptitle("Cell Survival Predictions vs True values", fontsize = 25, position = (0.5, 0.95))
    for i in range(0,predictions.shape[1]):    
        for j in range(0,2):
            n = 2*i + j
            if n >= predictions.shape[1]:
                continue
            
            axs[i,j].plot(t, predictions[:,n], label = f"x_{n+1} pred", color = "b")
            axs[i,j].plot(time_t*t_scale, base_data[:,n], label = f'x_{n+1}', color = "r")
            axs[i,j].legend()

#%% Load Weights for model trained for Cell Death
os.chdir(r"C:\Users\danie\repos\INF399\trained_models")
models = [MLP(8,8,3) for i in range(0,8)]
for i, model in enumerate(models, start = 1):
    model.load_state_dict(torch.load("cd_weights" + str(i) + ".pth"))
#%%
cell_death_init = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e3, 0]
predictions = test_models(models, cell_death_init, 8)
plot_predictions(predictions)
#%% Load Weights for model trained for Cell Survival
os.chdir(r"C:\Users\danie\repos\INF399\trained_models")
models = [MLP(8,8,3) for i in range(0,8)]
for i, model in enumerate(models, start = 1):
    model.load_state_dict(torch.load("cs_weights" + str(i) + ".pth"))
#%%
cell_survival_init = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e4, 0]
predictions = test_models(models, cell_survival_init, 8)
plot_predictions(predictions)
#%% Load weights for Crauste Model
os.chdir(r"C:\Users\danie\repos\INF399\trained_models")
models = [MLP(5,8,3) for i in range(0,5)]
for i, model in enumerate(models, start = 1):
    model.load_state_dict(torch.load("craustebase_weights" + str(i) + ".pth"))
#%%
crauste_init = [8090,0,0,0,1]
predictions = test_models(models, crauste_init, 5)
plot_predictions(predictions)
#%%
print(predictions[0,:])

#%%
x = [3000,800,110,10,100]
print(models[3](torch.tensor(x, dtype=torch.float32)))
