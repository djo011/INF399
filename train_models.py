# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:40:02 2023

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

def train(model, model_number, dataloader, epochs, criterion, optimizer):
    total_train_loss = []
    total_val_loss = []
    for epoch in range(1,epochs + 1):
        train_loss = []
        val_loss = []
        # Training loop
        for i, data in enumerate(dataloader[0], 0):
            model.train()
            # get input to train model, and true values
            x, true  = data
            # print(x.shape)
            #Zero the parameter gradients
            optimizer.zero_grad()
            
            #Forward and make prediction, calculate loss
            prediction = model(x.float())
            loss = criterion(prediction, true[:,model_number].unsqueeze(1))
            loss.backward()
            
            #Backward
            optimizer.step()
            
            #Update current loss
            loss = loss.item()
            train_loss.append(loss)
            
        #Validation loop   
        for i, data in enumerate(dataloader[1], 0):
            model.eval()
            x_val, true_val = data
            val_pred = model(x_val.float())
            vloss = criterion(val_pred, true_val[:,model_number].unsqueeze(1))
            
            val_loss.append(vloss.item())
            
        
        print(f'Epoch {epoch}/{epochs} done for model number {model_number+1}')
        print(f'Average training loss = {sum(train_loss)/len(dataloader[0].dataset)}')
        print(f'Average validation loss = {sum(val_loss)/len(dataloader[1].dataset)}')
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)
        
    return [total_train_loss, total_val_loss]

def train_models(models, epochs, dataloader):
    train_loss = []
    val_loss = []
    for n in range(0,len(models)):
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(models[n].parameters(), lr=1e-3)
        model_loss = train(models[n], n, dataloader, epochs, criterion, optimizer)
        print(f'Model number {n+1} done training!')
        train_loss.append(model_loss[0])
        val_loss.append(model_loss[1])
        
    return [train_loss,val_loss]

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
    data_gen = CellApoModel(cell_survival=False)
    base_data = data_gen.base_data
    time_t = data_gen.time_t
    t_scale = data_gen.t_scale
    t = [i for i in range(0,60)]
    fig, axs = plt.subplots(math.ceil(predictions.shape[1]/2),2, figsize = (15,15))
    fig.suptitle("Cell Death Predictions vs True values", fontsize = 25, position = (0.5, 0.95))
    for i in range(0,predictions.shape[1]):    
        for j in range(0,2):
            n = 2*i + j
            if n >= predictions.shape[1]:
                continue
            
            axs[i,j].plot(t, predictions[:,n], label = f"x_{n+1} pred", color = "b")
            axs[i,j].plot(time_t*t_scale, base_data[:,n], label = f'x_{n+1}', color = "r")
            axs[i,j].legend()

#%% Generate Training Data
noise_lvl = 0.05
training_set_size = 10
data_points = 60
data_generator = CrausteModel()
data_generator.gen_data(noise_lvl,training_set_size,data_points, "frequenttimeframe")
data = data_generator.get_data_set()
data_generator.plot_data()
X = data[:,:,:-1,0].transpose(0,2,1) # dim(data_set, species, data_points)
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]) # dim (data_set * data_points, species)
Y = data[:,:,1:,0].transpose(0,2,1)
Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])

np.save(r"C:\Users\danie\repos\INF399\data\training_set_crauste.npy", (X,Y))
#%% Generate Validation Data
noise_lvl = 0.05
validation_set_size = 10
data_points = 60
data_generator = CrausteModel()
data_generator.gen_data(noise_lvl,validation_set_size,data_points, "frequenttimeframe")
data = data_generator.get_data_set()

X = data[:,:,:-1,0].transpose(0,2,1) # dim(data_set, species, data_points)
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]) # dim (data_set * data_points, species)
Y = data[:,:,1:,0].transpose(0,2,1)
Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])

np.save(r"C:\Users\danie\repos\INF399\data\validation_set_crauste.npy", (X,Y))

#%% Load data
os.chdir(r"C:\Users\danie\repos\INF399\data")
training_set = GeneExpressionData("training_set_crauste.npy")
validation_set = GeneExpressionData("validation_set_crauste.npy")
TrainLoader = DataLoader(training_set, batch_size = 4, shuffle = True)
ValidationLoader = DataLoader(validation_set, batch_size = 4, shuffle = True)

dataloader = [TrainLoader,ValidationLoader]
#%% Train models
epochs = 100
models = [MLP(5,64,5) for i in range(0,5)]
loss_for_model = train_models(models, epochs, dataloader)

#%% Save models
os.chdir(r"C:\Users\danie\repos\INF399\trained_models")
for i in range(0,5):
    torch.save(models[i].state_dict(), "crauste4_weights" + str(i+1) +".pth")
    
    
#%% Generate Training Data CELL SURVIVAL
noise_lvl = 0.05
training_set_size = 500
data_points = 60
data_generator = CellApoModel(cell_survival = True)
data_generator.gen_data(noise_lvl,training_set_size,data_points)
data = data_generator.get_data_set()

X = data[:,:,:-1,0].transpose(0,2,1) # dim(data_set, species, data_points)
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]) # dim (data_set * data_points, species)
Y = data[:,:,1:,0].transpose(0,2,1)
Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])

np.save(r"C:\Users\danie\repos\INF399\data\training_set_cs.npy", (X,Y))
#%% Generate Validation Data
noise_lvl = 0.05
validation_set_size = 50
data_points = 60
data_generator = CellApoModel(cell_survival = True)
data_generator.gen_data(noise_lvl,validation_set_size,data_points)
data = data_generator.get_data_set()

X = data[:,:,:-1,0].transpose(0,2,1) # dim(data_set, species, data_points)
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]) # dim (data_set * data_points, species)
Y = data[:,:,1:,0].transpose(0,2,1)
Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])

np.save(r"C:\Users\danie\repos\INF399\data\validation_set_cs.npy", (X,Y))

#%% Load data
os.chdir(r"C:\Users\danie\repos\INF399\data")
training_set = GeneExpressionData("training_set_cs.npy")
validation_set = GeneExpressionData("validation_set_cs.npy")
TrainLoader = DataLoader(training_set, batch_size = 4, shuffle = True)
ValidationLoader = DataLoader(validation_set, batch_size = 4, shuffle = True)

dataloader = [TrainLoader,ValidationLoader]
#%% Train models
epochs = 20
models = [MLP(8,8,3) for i in range(0,8)]
loss_for_model = train_models(models, epochs, dataloader)

#%% Save models
os.chdir(r"C:\Users\danie\repos\INF399\trained_models")
for i in range(0,8):
    torch.save(models[i].state_dict(), "cs_weights" + str(i+1) +".pth")
    
#%% Test models
import math

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
    data_gen = CellApoModel(cell_survival=True)
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

cell_survival_init = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e4, 0]
predictions = test_models(models, cell_survival_init, 8)
plot_predictions(predictions)

#%% Generate Training Data CELL DEATH
noise_lvl = 0.05
training_set_size = 500
data_points = 60
data_generator = CellApoModel(cell_survival = False)
data_generator.gen_data(noise_lvl,training_set_size,data_points)
data = data_generator.get_data_set()

X = data[:,:,:-1,0].transpose(0,2,1) # dim(data_set, species, data_points)
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]) # dim (data_set * data_points, species)
Y = data[:,:,1:,0].transpose(0,2,1)
Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])

np.save(r"C:\Users\danie\repos\INF399\data\training_set_cd.npy", (X,Y))
#%% Generate Validation Data
noise_lvl = 0.05
validation_set_size = 50
data_points = 60
data_generator = CellApoModel(cell_survival = False)
data_generator.gen_data(noise_lvl,validation_set_size,data_points)
data = data_generator.get_data_set()

X = data[:,:,:-1,0].transpose(0,2,1) # dim(data_set, species, data_points)
X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]) # dim (data_set * data_points, species)
Y = data[:,:,1:,0].transpose(0,2,1)
Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])

np.save(r"C:\Users\danie\repos\INF399\data\validation_set_cd.npy", (X,Y))

#%% Load data
os.chdir(r"C:\Users\danie\repos\INF399\data")
training_set = GeneExpressionData("training_set_cd.npy")
validation_set = GeneExpressionData("validation_set_cd.npy")
TrainLoader = DataLoader(training_set, batch_size = 4, shuffle = True)
ValidationLoader = DataLoader(validation_set, batch_size = 4, shuffle = True)

dataloader = [TrainLoader,ValidationLoader]
#%% Train models
epochs = 20
models = [MLP(8,8,3) for i in range(0,8)]
loss_for_model = train_models(models, epochs, dataloader)

#%% Save models
os.chdir(r"C:\Users\danie\repos\INF399\trained_models")
for i in range(0,8):
    torch.save(models[i].state_dict(), "cd_weights" + str(i+1) +".pth")
#%%
cell_death_init = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e3, 0]
predictions = test_models(models, cell_death_init, 8)
plot_predictions(predictions)

#%%
crauste_gen = CrausteModel()
crauste_base = crauste_gen.base_data
X_base = crauste_base[:200,:]
Y_base = crauste_base[1:201,:]

np.save(r"C:\Users\danie\repos\INF399\data\training_set_craustebase.npy", (X_base, Y_base))
np.save(r"C:\Users\danie\repos\INF399\data\validation_set_craustebase.npy", (X_base, Y_base))
#%%
os.chdir(r"C:\Users\danie\repos\INF399\data")
training_set = GeneExpressionData("training_set_craustebase.npy")
validation_set = GeneExpressionData("validation_set_craustebase.npy")
TrainLoader = DataLoader(training_set, batch_size = 4, shuffle = True)
ValidationLoader = DataLoader(validation_set, batch_size = 4, shuffle = True)

dataloader = [TrainLoader,ValidationLoader]
epochs = 150
models = [MLP(5,8,3) for i in range(0,5)]

loss_for_model = train_models(models, epochs, dataloader)

#%% Save models
os.chdir(r"C:\Users\danie\repos\INF399\trained_models")
for i in range(0,5):
    torch.save(models[i].state_dict(), "craustebase_weights" + str(i+1) +".pth")
