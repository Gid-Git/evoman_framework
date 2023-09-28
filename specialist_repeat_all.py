from BitFlip_3point_Elitism import run_evoman as run_evoman_bitflip_3point_elitism
from BitFlip_3point_Tournament import run_evoman as run_evoman_bitfplit_3point_tournament
from BitFlip_Uniform_Elitism import run_evoman as run_evoman_bitflip_uniform_elitism
from BitFlip_Uniform_Tournament import run_evoman as run_evoman_bitflip_uniform_tournament
from Gaussian_3point_Elitism import run_evoman as run_evoman_gaussian_3point_elitism
from Gaussian_3point_Tournament import run_evoman as run_evoman_gaussian_3point_tournament
from Gaussian_Uniform_Elitism import run_evoman as run_evoman_gaussian_uniform_elitism
from Gaussian_Uniform_Tournament import run_evoman as run_evoman_gaussian_uniform_tournament
import time
import numpy as np
import os 
import csv

parameters = {
    'experiment_name': "",
    'enemy': [6],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.1,
    'crossover_rate': 0.5,
    'mode': "train",
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
    'number_of_crossovers': 3,
    'n_elitism': 0,
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
        parameters['experiment_name'] = EA[0] + f'run2_{i}'
        runs_best_fitness[i] = EA[1](**parameters)    

    print()
    print(f"EA used:", EA[0])
    print(f"Average best achieved fitness over {number_of_runs} runs:", np.average(runs_best_fitness))
    print(f"Average best achieved fitness over {number_of_runs} runs:", np.std(runs_best_fitness))
    print()
