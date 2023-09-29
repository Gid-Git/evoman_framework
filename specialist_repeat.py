from BitFlip_3point_Elitism import run_evoman
import time

parameters = {
    'experiment_name': "",
    'enemy': [6],
    'population_size': 100,
    'generations': 30,
    'mutation_rate': 0.07986069693792121,
    'crossover_rate': 0.9317683828391756,
    'mode': "train",
    'n_hidden_neurons': 7,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
    'number_of_crossovers': 8,
    'n_elitism': 16,
    'k_tournament': 17,
    'sel_pres_incr': True,
    'k_tournament_final_linear_increase_factor': 3,   
}


# for running and saving the code 10 times
for i in range(10):
    number = i + 1
    unique_experiment_name = f"tuned_experiment_{number}"
    parameters["experiment_name"] = unique_experiment_name
    run_evoman(**parameters)