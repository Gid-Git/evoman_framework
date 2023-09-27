from bitflip_3point_elitism import run_evoman as run_evoman_bitflip_3point_elitism
from bitflip_3point_tournament import run_evoman as run_evoman_bitfplit_3point_tournament
from bitflip_uniform_elitism import run_evoman as run_evoman_bitflip_uniform_elitism
from bitflip_uniform_tournament import run_evoman as run_evoman_bitflip_uniform_tournament
from gaussian_3point_elitism import run_evoman as run_evoman_gaussian_3point_elitism
from gaussian_3point_tournament import run_evoman as run_evoman_gaussian_3point_tournament
from gaussian_uniform_elitism import run_evoman as run_evoman_gaussian_uniform_elitism
from gaussian_uniform_tournament import run_evoman as run_evoman_gaussian_uniform_tournament
import time
import numpy as np
import os 
import csv

parameters = {
    'experiment_name': "",
    'enemy': [6],
    'population_size': 100,
    'generations': 50,
    'mutation_rate': 0.1,
    'crossover_rate': 0.9,
    'mode': "train",
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
    'number_of_crossovers': 3,
    'n_elitism': np.inf,
    'k_tournament': 4,
    'sel_pres_incr': True,
    'k_tournament_final_linear_increase_factor': 4,   
}

EAS = {
    'Bitflip-3Point-Elitism': run_evoman_bitflip_3point_elitism,
    'Bitfplit-3Point-Tournament': run_evoman_bitfplit_3point_tournament,
    'Bitflip-Uniform-Elitism': run_evoman_bitflip_uniform_elitism,
    'Bitflip-Uniform-Tournament': run_evoman_bitflip_uniform_tournament,
    'Gaussian-3Point-Elitism': run_evoman_gaussian_3point_elitism,
    'Gaussian-3Point-Tournament': run_evoman_gaussian_3point_tournament,
    'Gaussian-Uniform-Elitism': run_evoman_gaussian_uniform_elitism,
    'Gaussian-Uniform-Tournament': run_evoman_gaussian_uniform_tournament,
    }

number_of_runs = 10

for EA in EAS.items():
    runs_best_fitness = np.zeros(number_of_runs)
    #Check if elitism, if so set elitism to two
    if EA[0][-1] == 'm':
        parameters['n_elitism'] = 2
    for i in range(number_of_runs):
        parameters['experiment_name'] = EA[0] + f'run_{i}'
        runs_best_fitness[i] = EA[1](**parameters)    

    print()
    print(f"EA used:", EA[0])
    print(f"Average best achieved fitness over {number_of_runs} runs:", np.average(runs_best_fitness))
    print(f"Average best achieved fitness over {number_of_runs} runs:", np.std(runs_best_fitness))
    print()
