import numpy as np
from pyDOE2 import lhs
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from new_specialist_last import run_evoman
import csv

# Define fixed parameters and parameters to change
parameters = {
    'experiment_name': "Parameter_Tuning",
    'enemy': [6],
    'population_size': 100,
    'generations': 2,
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

param_change = {
    'mutation_rate': {'range': (0.01, 0.2), 'type': 'float'},
    'crossover_rate': {'range': (0.2, 1), 'type': 'float'},
    'dom_l': {'range': (-1, 1), 'type': 'float'},
    'dom_u': {'range': (-1, 1), 'type': 'float'},
    'generations': {'range': (1, 50), 'type': 'int'}
}


n_samples = 50
n_iterations = 50

# Initialize lists for storing evaluated parameters and performances
evaluated_parameters = []
performances = []

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

    performance = run_evoman(**combined_parameters)
    initial_performance.append(performance)

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
    next_performance = run_evoman(**combined_parameters)

    evaluated_parameters.append(varying_parameters)
    performances.append(next_performance)
    model.fit(np.vstack([model.X_train_, res.x]), np.hstack([model.y_train_, next_performance]))

# Find the best parameters and save results
best_index = np.argmax(performances)
best_parameters = evaluated_parameters[best_index]

with open('evaluated_parameters.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(list(param_change.keys()) + ['Performance'])
    for params, performance in zip(evaluated_parameters, performances):
        writer.writerow(list(params.values()) + [performance])

with open('best_parameters.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(best_parameters.keys())
    writer.writerow(best_parameters.values())

print(f"Best Parameters: {best_parameters}")