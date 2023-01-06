# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:13:04 2022

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import argparse
import time
import math

class BaseModel():
    def __init__(self):
        self.model_name = None
        self.init_val = None
        self.time_t = None
        self.t_scale = None
        self.x_scale = None
        self.data_set = None # dim = (n_data_set, n_species, n_data_points, 2)
        self.base_data = None
        self.n_species = None
        
    def get_data_set(self):
        return self.data_set
    
    def ode_model(self):
        return 0
    
    def gen_data(self, noise_lvl, n_data_set, n_data_points, random_sampling = False):
        start_time = time.time()
        
        # Store Original solution to ODE Model
        base_data = odeint(self.ode_model, self.init_val, self.time_t) # dim=(time_points ,n_species)
        self.base_data = base_data
        data = np.zeros((n_data_set, self.n_species, n_data_points, 3)) # Store both value and time for each species
        
        # Calculate std along each species.
        noise = base_data.std(0)*noise_lvl # dim = (n_species,)
        rng = np.random.default_rng()
        
        for n in range(n_data_set):
            temp_data = np.array(
                [rng.normal(0, noise[i], base_data.shape[0]) for i in range(self.n_species)]) # dim = (n_species, time_points)
            temp_data = temp_data.transpose() + base_data # dim(time_points, n_species)
            if random_sampling:    
                t_idx = rng.choice(base_data.shape[0], n_data_points,replace=False) # dim = (n_data_points,)
            else:
                t_idx = np.linspace(0,base_data.shape[0]-1, n_data_points, dtype= np.int32)
                
            temp_t = self.time_t[t_idx] # dim = (n_data_points)
            temp_data = temp_data[t_idx,:].transpose()
            temp_data = np.where(temp_data<0, 0, temp_data)
            vel = np.array([self.ode_model(temp_data[:,i], temp_t[i]) for i in range(0,n_data_points)]).transpose()
            data[n,:,:,0] = temp_data
            data[n,:,:,1] = temp_t
            data[n,:,:,2] = vel
            
            
            if n % 100 == 0:
                print(f'Generated {n}/{n_data_set} datasets')
        
        
        end_time = time.time()
        print(f'Data generated in: {end_time-start_time}s')
        self.data_set = data
    
    def plot_data(self):
        max_i = math.ceil(self.n_species/2)
        max_j = 2
        fig, axs = plt.subplots(max_i,max_j)
        for i in range(0,max_i):
            for j in range(0,max_j):
                n = 2*i + j
                if n >= self.n_species:
                    continue
                axs[i,j].scatter(self.data_set[0,n,:,1]*self.t_scale,self.data_set[0,n,:,0]*self.x_scale, color = "blue")
                axs[i,j].plot(self.time_t*self.t_scale, self.base_data[:,n]*self.x_scale, color = "red")
                #axs[i,j].xlabel(self.x_label)
                #axs[i,j].ylabel(self.y_label)
                #axs[i,j].set_title(f'x_{n+1} ' + self.plot_title)
        fig.suptitle(self.plot_title)
    
    def plot_basedata(self):
        max_i = math.ceil(self.n_species/2)
        max_j = 2
        fig, axs = plt.subplots(max_i,max_j)
        fig.suptitle("Base data for" + self.model_name)
        for i in range(0,max_i):
            for j in range(0,max_j):
                n = 2*i + j
                if n >= self.n_species:
                    continue
                axs[i,j].plot(self.time_t*self.t_scale, self.base_data[:,n]*self.x_scale)
        """
        for n in range(0,self.n_species):
            plt.plot(self.time_t*self.t_scale, self.base_data[:,n]*self.x_scale, label = f"x_{n+1}")
        plt.title("Base data for" + self.model_name)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        """

class CellApoModel(BaseModel):
    def __init__(self, cell_survival = False):
        
        if cell_survival:
            self.model_name = "Cell Survival"
            self.init_val = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e4, 0]
            self.plot_title = " Expression in Cell Survival"
        
        else:
            self.model_name = "Cell Apotosis"
            self.init_val = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e3, 0]
            self.plot_title = " Expression in Cell Apotosis"
            
        self.time_t = np.linspace(0,60**3,60**3)
        self.t_scale = 1/3600
        self.x_scale = 1/1e5
        self.data_set = None
        self.data_time_set = None
        self.x_label = "t(h)"
        self.y_label = "Molecules pr cell (10^5)"
        self.n_species = 8
        self.base_data = odeint(self.ode_model, self.init_val, self.time_t)
        
    def ode_model(self, x, t):
        """
        
        Function Description
        --------------------
        
        Function will return derivative values for biochemical species x at time t.
        
        Returns
        -------
        
        dxdt : List of float
            The change in level of biochemical species.

        """
        
        #Init parameters in ODE's
        k = {"1": 2.67e-9,
             "2": 0,
             "3": 6.8e-8,
             "4": 0,
             "5": 7e-5}
        kd = {"1": 1e-2,
              "2": 8e-3,
              "3": 5e-2,
              "4": 1e-3,
              "5": 1.67e-5,
              "6": 1.67e-4}
        
        dx_1dt = -k["1"]*x[3]*x[0] + kd["1"]*x[4]
        dx_2dt = kd["2"]*x[4] - k["3"]*x[1]*x[2] + kd["3"]*x[5] + kd["4"]*x[5]
        dx_3dt = -k["3"]*x[1]*x[2] + kd["3"]*x[5]
        dx_4dt = kd["4"]*x[5] - k["1"]*x[3]*x[0] + kd["1"]*x[4] - k["5"]*x[6]*x[3] + kd["5"]*x[7] + kd["2"]*x[4]
        dx_5dt = -kd["2"]*x[4] + k["1"]*x[3]*x[0] - kd["1"]*x[4]
        dx_6dt = -kd["4"]*x[5] + k["3"]*x[1]*x[2] - kd["3"]*x[5]
        dx_7dt = -k["5"]*x[6]*x[3] + kd["5"]*x[7] + kd["6"]*x[7]
        dx_8dt = k["5"]*x[6]*x[3] - kd["5"]*x[7] - kd["6"]*x[7]
        
        dxdt = [dx_1dt, dx_2dt, dx_3dt, dx_4dt, dx_5dt, dx_6dt, dx_7dt, dx_8dt]
        
        return dxdt
    
    
class CrausteModel(BaseModel):
    def __init__(self, init_val = [8090,0,0,0,1]):
        self.model_name = "Crauste Model"
        self.init_val = init_val
        self.time_t = np.linspace(0,60,601)
        self.t_scale = 1/1
        self.x_scale = 1/1
        self.data_set = None
        self.data_time_set = None
        self.x_label = "t(h)"
        self.y_label = "Concentration (pM)"
        self.plot_title = " Expression in Crauste Model"
        self.n_species = 5
        self.base_data = odeint(self.ode_model, self.init_val, self.time_t)
    
    def ode_model(self,x, t):  
        
        mu = {
            "N" : 0.739907308603256,
            "EE" : 3.91359322673521*1e-5,
            "EL" : 0,
            "E" : 0,
            "LL" : 8.11520135326853*1e-6,
            "LE" : 1.00000000000005*1e-10,
            "L" : 0,
            "M" : 0,
            "PE": 1.36571832778378*1e-10,
            "PL": 3.6340308186265*1e-5,
            "P" : 1.00000002976846*1e-5} 
        delta = {
            "NE": 0.0119307857579241,
            "EL" : 0.51794597529254,
            "LM" : 0.0225806365892933}
        rho = {
            "E" : 0.507415703707752,
            "P" : 0.126382288121756}
        
        dNdt = - mu["N"]*x[0] - delta["NE"]*x[4]*x[0]
        dEdt = delta["NE"]*x[4]*x[0] + (rho["E"]*x[4] - mu["EE"]*x[1] - mu["EL"]*x[2] - mu["E"] - delta["EL"])*x[1]
        dLdt = delta["EL"]*x[1] - (mu["LL"]*x[2] + mu["LE"]*x[1] + mu["L"] + delta["LM"])*x[2]
        dMdt = delta["LM"]*x[2] - mu["M"]*x[3]
        dPdt =(rho["P"]*x[4] - mu["PE"]*x[1] - mu["PL"]*x[2] - mu["P"])*x[4]
        
        dxdt = [dNdt, dEdt, dLdt, dMdt, dPdt]
        
        return dxdt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generate simulated dataset for given biological function')
    
    parser.add_argument('--model-name', type = str, default='crauste',
                        help='Name of model used to simulate data(cellapotosis or crauste')
    parser.add_argument('--n-data-set', type = int, default= 100,
                        help='Number of datasets to be simulated')
    parser.add_argument('--noise-level', type = float, default = 0.05,
                        help="The noise level of simulated data")
    parser.add_argument('--n-data-points', type = int, default = 60,
                        help="The number of datapoints in the sampled trajectory. Default is 60, which equals one sample every hour")
    parser.add_argument('--random-sampling', type = bool, default = False,
                        help='When true, sample randomly in timeframe. When False sample evenly througout timeframe')
    
    args = parser.parse_args()
    
    
    if args.model_name == "crauste":
        model = CrausteModel()
        
        model.gen_data(args.noise_level, args.n_data_set, args.n_data_points)
        
        data_set = model.get_data_set() # dim(n_dat_set,n_species,n_data_points,(concentration,time,change of concentration))
        np.save("crauste_dataset.npy", data_set)
        
                
    if args.model_name == "cellapotosis":
        model_d = CellApoModel()
        model_s = CellApoModel(cell_survival = True)
        
        model_d.gen_data(args.noise_level, int(args.n_data_set/2), args.n_data_points)
        model_s.gen_data(args.noise_level, int(args.n_data_set/2), args.n_data_points)
        
        data_survival = model_s.get_data_set() 
        data_death = model_d.get_data_set()
        data_set = np.zeros((args.n_data_set,8,args.n_data_points,3)) # dim(n_dat_set,n_species,n_data_points,(concentration,time,change of concentration))
        data_set[:int(args.n_data_set/2),:,:] = data_survival[:,:,:,:]
        data_set[int(args.n_data_set/2):,:,:] = data_death[:,:,:,:]
        np.random.shuffle(data_set)
        np.save("cellapotosis_dataset.npy", data_set)
        