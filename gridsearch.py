from optimization_mick_2 import run_evoman

parameters = {
    'experiment_name': "grid_search",
    'enemy': [6],
    'population_size': 100,
    'generations': 50,
    'mutation_rate': 0.1,
    'crossover_rate': 0.9,
    'tournament_size': 5,
    'mode': "train",
    'n_hidden_neurons': 10,
    'headless': True,
    'dom_l': -1,
    'dom_u': 1,
    'speed': "fastest", 
}
# in this case I choose it to be mutation, You can change everything here
grid = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# grid search
for value in grid:
    unique_experiment_name = f"mutation_grid_{value}"
    parameters["experiment_name"] = unique_experiment_name
    run_evoman(**parameters)
