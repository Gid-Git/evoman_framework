import numpy as np
from pyDOE2 import lhs
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from new_specialist_last import run_evoman
import csv


parameters = {
    'experiment_name': "testing_specialistx10",
    'enemy': [6],
    'population_size': 100,
    'generations': 30,
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

"""
Dont forget to later try this with the different selection, mutation and turnament
"""

# If we have more time later add dom_l and dom_u as well as tournament size and so on look at lower function
param_change = {
    'mutation_rate': (0.01, 0.2),
    'crossover_rate': (0.2, 1),
    'dom_l': (-1, 1),
    'dom_u': (-1, 1)
}

"""Add this when doing with n_hidden_neurons"""

# # Find the next point to sample
# def objective(x):
#     # Round the value of n_hidden_neurons to the nearest integer
#     x[-1] = round(x[-1])
#     return -model.predict([x])[0]  # We want to maximize the performance, hence minimize the negative performance
    
# res = minimize(objective, [0.5] * len(ranges), bounds=normalized_ranges)
# next_point = res.x * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    
# # Round the number of hidden neurons to the nearest integer
# next_point[-1] = round(next_point[-1])


n_samples = 100
n_iterations = 100

evaluated_parameters = []
performances = []

# Normalize ranges
ranges = np.array(list(param_change.values()))
normalized_ranges = np.array([[0, 1]] * len(ranges))

# Generate initial design points using lhs
initial_design = lhs(len(ranges), samples=n_samples)
scaled_design = initial_design * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

# Evaluate the initial design points using run_evoman
initial_performance = []
for point in scaled_design:
    varying_parameters = dict(zip(param_change.keys(), point))
    combined_parameters = {**parameters, **varying_parameters}
    performance = run_evoman(**combined_parameters)
    initial_performance.append(performance)

initial_performance = np.array(initial_performance)

# Fit the Gaussian Process Regression model
model = GaussianProcessRegressor()
model.fit(initial_design, initial_performance)

for iteration in range(n_iterations):
    # Find the next point to sample
    def objective(x):
        return -model.predict([x])[0]  # We want to maximize the performance, hence minimize the negative performance
    
    res = minimize(objective, [0.5] * len(ranges), bounds=normalized_ranges)
    next_point = res.x * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    varying_parameters = dict(zip(param_change.keys(), next_point))
    combined_parameters = {**parameters, **varying_parameters}

    next_performance = run_evoman(**combined_parameters)

    evaluated_parameters.append(dict(zip(param_change.keys(), next_point)))
    performances.append(next_performance)
    
    # Update the model with the new point
    model.fit(np.vstack([model.X_train_, res.x]), np.hstack([model.y_train_, next_performance]))

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