# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:35:52 2022

@author: danie
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import math

os.chdir(r"C:\Users\danie\repos\INF399")
os.chdir(r".\utils")
from NeuralNetwork import MLP
os.chdir(r"..\dataGeneration")
from simdata_generation import CrausteModel

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
    fig.suptitle("Crauste Predictions vs True values", fontsize = 25, position = (0.5, 0.95))
    for i in range(0,predictions.shape[1]):    
        for j in range(0,2):
            n = 2*i + j
            if n >= predictions.shape[1]:
                continue
            
            axs[i,j].plot(t, predictions[:,n], label = f"x_{n+1} pred", color = "b")
            axs[i,j].plot(time_t*t_scale, base_data[:,n], label = f'x_{n+1}', color = "r")
            axs[i,j].legend()

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= "Train models for Crauste model")
    parser.add_argument('--hidden_layer_size', type = int, default = 12, 
                        help='The number of nodes in each hidden layer')
    parser.add_argument('--n_hidden_layers', type = int, default = 3,
                        help='The number of hidden layers in MLP')
    args = parser.parse_args()
    hidden_layer_size = args.hidden_layer_size
    n_hidden_layers = args.n_hidden_layers
    
    os.chdir(r"C:\Users\danie\repos\INF399\trained_models\crauste")
    n_species = 5
    models = [MLP(n_species,hidden_layer_size, n_hidden_layers) for n in range(0,n_species)]
    
    for i, model in enumerate(models,start = 1):
        model.load_state_dict(torch.load("Crauste_model"+str(i)+"_weights.pth"))
    
    init_val = [8090,0,0,0,10]
    predictions = test_models(models, init_val, n_species)
    plot_predictions(predictions)