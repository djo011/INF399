# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:02:22 2022

@author: danie
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
os.chdir(r"C:\Users\danie\UiB\Master\Data_gen")
from NeuralNetwork import MLP
from simdata_generation import CrausteModel

class GeneExpressionData(Dataset):
    def __init__(self, filename):
        self.data = np.load(filename)
        
        self.X = torch.tensor(self.data[0]).float()
        self.Y = torch.tensor(self.data[1]).float()
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def gen_data(noise_lvl, size, save_data):
    data_generator = CrausteModel()
    data_generator.gen_data(noise_lvl, size, 60)
    
    data = data_generator.get_data_set()
    
    X = data[:,:,:-1,0].transpose(0,2,1) # dim(data_set, species, data_points)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2]) # dim (data_set * data_points, species)
    Y = data[:,:,1:,0].transpose(0,2,1)
    Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])
        
    train_val_partition = int(0.9*X.shape[0])
    X_train = X[:train_val_partition]
    X_val = X[train_val_partition:]
    Y_train = Y[:train_val_partition]
    Y_val = Y[train_val_partition:]
    
    train_set = (X_train,Y_train)
    val_set = (X_val,Y_val)
    
    if save_data:
        np.save('data\crauste_training_set.npy', train_set)
        np.save('data\crauste_validation_set.npy', val_set)


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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= "Train models for Crauste model")
    
    parser.add_argument('--size_dataset', type = int, default = 1000,
                        help = 'The size of dataset, one dataset equals 60 data points')
    parser.add_argument('--noise-level', type = float, default = 0.05,
                        help = 'The level of noise in generated data')
    parser.add_argument('--save_data',type = bool, default = True,
                        help = 'Determine if data should be save or not')
    parser.add_argument('--epochs', type = int, default = 15,
                        help='The number of epochs to be used in training')
    
    #Parse arguments and store in variables
    args = parser.parse_args()
    size = args.size_dataset
    noise_lvl = args.noise_level
    save_data = args.save_data
    epochs = args.epochs
    
    start = time.time()
    gen_data(noise_lvl, size, save_data)
    
    training_data = GeneExpressionData('data\crauste_training_set.npy')
    validation_data = GeneExpressionData('data\crauste_validation_set.npy')
    
    train_loader = DataLoader(training_data)
    val_loader = DataLoader(validation_data)
    
    Crauste_models = [MLP(5,8) for n in range(0,5)]
    
    training_loss, validation_loss = train_models(Crauste_models, epochs, [train_loader, val_loader])
    
    os.chdir(r"C:\Users\danie\UiB\Master\Data_gen\trained_models\crauste")
    for i in range(len(Crauste_models)):
        torch.save(Crauste_models[i].state_dict(), f'Crauste_model{i+1}_weights0812.pth')
    
    np.save('crauste_trainingloss.npy', training_loss)
    np.save('crauste_validationloss.npy', validation_loss)    
    end = time.time()
    
    print(f'Models Trained in {(end-start)/60} min(s)')