from scipy.integrate import odeint
from scipy.stats import qmc
import numpy as np

def lh_sampling(number_of_samples, n_species):
    """
    Function will do Latin HyperCube sampling for initial values of Cell Apotosis function.
    ---------------------------------------------------------------------------------------
    Input:
    int number_of_samples: The number of samples to draw
    int n_species: The number of species in the Cell Apotosis function
    Return:
    array drawn_samples: An array with the desired number of drawn samples.
    """    
    # init_val_cell_survival = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e4, 0]
    # init_val_cell_apotosis = [1.34e5, 1e5, 2.67e5, 0, 0, 0, 2.9e3, 0]
    l_bound = [1.34e4, 1e4, 2.67e4, 0, 0, 0, 2.9e2, 0]
    u_bound = [1.34e6, 1e6, 2.67e6, 1, 1, 1, 2.9e5, 1]
    drawn_samples = np.zeros((number_of_samples, n_species))
    lh_sampler = qmc.LatinHypercube(d=n_species)
    lh_sample = lh_sampler.random(n = number_of_samples)
    drawn_samples = qmc.scale(lh_sample, l_bound, u_bound)

    return drawn_samples

def test_lh_sampling():
    n_samples = 10
    n_species = 8
    l_bound = [1.34e4, 1e4, 2.67e4, 0, 0, 0, 2.9e2, 0]
    u_bound = [1.34e6, 1e6, 2.67e6, 1, 1, 1, 2.9e5, 1]
    test_samples = lh_sampling(n_samples, n_species)
    
    assert test_samples.shape[0] == n_samples
    assert test_samples.shape[1] == n_species

    for i in range(0,len(test_samples[0])):
        for j in range(0, len(test_samples[1])):
            sample = test_samples[i][j]
            if sample < l_bound[j]:
                print("Sample was out of bounds! Lower than desired!")
                return False
            if sample > u_bound[j]:
                print("Sample was out of bounds! Greater than desired!")
                return False
    print("All test passed!")
    return True 

test_lh_sampling()
