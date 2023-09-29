import numpy as np
from pyDOE2 import lhs
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from BitFlip_3point_Elitism import run_evoman
import csv

# Define fixed parameters and parameters to change
parameters = {
    'experiment_name': "",
    'enemy': [6],
    'population_size': 100,
    'generations': 5,
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

param_change = {
    'mutation_rate': {'range': (0.01, 0.2), 'type': 'float'},
    'crossover_rate': {'range': (0.2, 1), 'type': 'float'},
    'n_hidden_neurons': {'range': (2, 20), 'type': 'int'},
    'k_tournament': {'range': (2, 20), 'type': 'int'},
    'number_of_crossovers': {'range': (2, 20), 'type': 'int'},
    'n_elitism': {'range': (2, 20), 'type': 'int'},
    'k_tournament_final_linear_increase_factor': {'range': (2, 5), 'type': 'int'}
}


n_samples = 2
n_iterations = 2
n_runs = 2

# Initialize lists for storing evaluated parameters and performances
evaluated_parameters = []
performances = []
all_performances = []

# Normalize ranges and generate initial design points
ranges = np.array([param['range'] for param in param_change.values()])
normalized_ranges = np.array([[0, 1]] * len(ranges))
initial_design = lhs(len(ranges), samples=n_samples)
scaled_design = initial_design * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

# Evaluate initial design points and fit the model
initial_performance = []
for point in scaled_design:
    varying_parameters = dict(zip(param_change.keys(), point))
    combined_parameters = {**parameters, **varying_parameters}
    for param_name, param_properties in param_change.items():
        if param_properties['type'] == 'int':
            combined_parameters[param_name] = int(round(combined_parameters[param_name]))

    performance_runs = [run_evoman(**combined_parameters) for _ in range(n_runs)]
    average_performance = np.mean(performance_runs)

    all_performances.append(performance_runs)
    average_performance.append(average_performance)

model = GaussianProcessRegressor()
model.fit(initial_design, np.array(initial_performance))

# Optimize and update the model iteratively
for iteration in range(n_iterations):
    def objective(x):
        return -model.predict([x])[0]
    
    res = minimize(objective, [0.5] * len(ranges), bounds=normalized_ranges)
    next_point = res.x * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    varying_parameters = dict(zip(param_change.keys(), next_point))
    for param_name, param_properties in param_change.items():
        if param_properties['type'] == 'int':
            varying_parameters[param_name] = int(round(varying_parameters[param_name]))

    combined_parameters = {**parameters, **varying_parameters}
    performance_runs = [run_evoman(**combined_parameters) for _ in range(n_runs)]
    next_performance = np.mean(performance_runs)

    evaluated_parameters.append(varying_parameters)
    average_performances.append(next_performance)
    all_performances.append(performance_runs)
    model.fit(np.vstack([model.X_train_, res.x]), np.hstack([model.y_train_, next_performance]))


# Save average performances and individual performances to CSV files
with open('average_performances.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(list(param_change.keys()) + ['Average Performance'])
    for params, avg_performance in zip(evaluated_parameters, average_performances):
        writer.writerow(list(params.values()) + [avg_performance])

with open('all_performances.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(list(param_change.keys()) + ['Run ' + str(i) for i in range(1, n_runs + 1)])
    for params, performances in zip(evaluated_parameters, all_performances):
        writer.writerow(list(params.values()) + performances)

# Find the best parameters
best_index = np.argmax(average_performances)
best_parameters = evaluated_parameters[best_index]

with open('best_parameters.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(best_parameters.keys())
    writer.writerow(best_parameters.values())

print(f"Best Parameters: {best_parameters}")