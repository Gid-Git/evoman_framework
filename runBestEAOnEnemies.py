from besteEArun import run_evoman
import pandas as pd
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

parameters = {
    'experiment_name': "",
    'enemy': [],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.1,
    'crossover_rate': 0.9,
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
    'alpha': 0.9
}

enemies = [4, 6, 7]
number_of_runs = 10

def runEA(version):
    for enemy in enemies:
        print(version, enemy)
        parameters['enemy'] = [enemy]
        for i in range(number_of_runs):
            number = i + 1
            unique_experiment_name = f"{version}_{enemy}_{number}"
            parameters["experiment_name"] = unique_experiment_name
            run_evoman(**parameters)

def adjust_parameters(version):
    best_parameters_dir = os.path.join(script_dir, "Results_Tuning", "individuals")
    best_parameters_path = os.path.join(best_parameters_dir, f"best_parameters_{version}.csv")
    df = pd.read_csv(best_parameters_path)
    for parameter in list(df):
        parameters[parameter] = df.iloc[0][parameter]    
    # Force some values to int that are necessary for evoman
    for parameter in list(df)[2:-1]:
        parameters[parameter] = round(parameters[parameter])


version = 'withouth_fitness'
adjust_parameters(version)
runEA(version)

version = 'with_fitness'
adjust_parameters(version)
runEA(version)
