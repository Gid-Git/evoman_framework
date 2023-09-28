from BitFlip_3point_Elitism import run_evoman
import time

# Define fixed parameters and parameters to change
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
    'n_elitism': 2,
    'k_tournament': 4,
    'sel_pres_incr': True,
    'k_tournament_final_linear_increase_factor': 4,   
}

start_time = time.time()

fitness = run_evoman(**parameters)

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print(fitness)