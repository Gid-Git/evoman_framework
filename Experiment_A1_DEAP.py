import sys
import os
import random
import numpy as np
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from demo_controller import player_controller
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

# Set up the EvoMan environment
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'specialist_assignment1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  enemies=[1, 2, 3],
                  multiplemode="no",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini='yes')

env.state_to_log()  # checks environment state

# Define the genetic algorithm parameters
MU = 30  # size of parent population
LAMBDA = 45  # size of generated offsprings
cxpb = 0.6  # probability of crossover
mutpb = 0.4  # probability of mutating
ngen = 4  # number of generations
dom_u = 1
dom_l = -1
nrep = 10  # number of times the experiment is repeated

ini = time.time()  # sets time marker

# Create individuals and population
IND_SIZE = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# define individual, fitness and strategy as datatypes
creator.create("FitnessMax", base.Fitness, weights=(
    1.0,))  # positive weight since it is not an error function, better solution = higher fitness score
creator.create("Individual", list, fitness=creator.FitnessMax)  # strategy for mutation over individual, defined later
creator.create("Strategy", list, typecode="d")

def generate_individual(individual, IND_SIZE):
    # Initializes individual of size n and populates it with randomly initialized strategy vector for mutation
    individual = creator.Individual(random.uniform(dom_l, dom_u) for _ in range(IND_SIZE))
    return individual

def fitness(env, individual):
    # return fitness value of one run of the game for an individual/solution with weights x
    f, p, e, t = env.play(pcont=individual)
    return (f,)  # needs to return a tuple

toolbox = base.Toolbox()
toolbox.register("individual", generate_individual, creator.Individual, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness, env)

###### register operators #######

# Mating strategy (of the parents), e.g. blending crossover
toolbox.register("mate", tools.cxUniform, indpb=0.05)
# Mutation strategy
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
# Selection strategy : Tournament selection
toolbox.register("select", tools.selTournament, tournsize=6)

# Define statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Data collection for plotting
avg_fitness_data = []
max_fitness_data = []

# Initialize the enemies for training
enemies = [1, 2, 3]

best_of_gens = np.zeros((nrep, 265))  # keep best solution from each generation to test

for enemy in enemies:
    env.update_parameter('enemies', [enemy])
    print('-------------- TRAINING AGAINST ENEMY {}--------------'.format(enemy))
    avg_fitness_over_reps = []
    max_fitness_over_reps = []

    for rep in range(nrep):
        print('---------- TRAINING REPETITION # {}----------'.format(rep + 1))
        hof = tools.HallOfFame(1)

        # Initialize the population
        population = toolbox.population(n=MU)
        final_population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=MU, lambda_=LAMBDA,
                                                              cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                                              halloffame=hof, stats=stats, verbose=False)

        # Store best solution from each repetition
        top_ind = hof[0]

        # Gather statistics for plotting
        avg_fitness = logbook.select("avg")
        max_fitness = logbook.select("max")
        avg_fitness_over_reps.append(avg_fitness)
        max_fitness_over_reps.append(max_fitness)

        # Store the best individual from each repetition
        best_of_gens[rep, :] = top_ind

    # Calculate and store the mean fitness values across repetitions for plotting
    avg_fitness_data.append(np.mean(avg_fitness_over_reps, axis=0))
    max_fitness_data.append(np.mean(max_fitness_over_reps, axis=0))

# Plot average and maximum fitness over generations for each enemy
for i, enemy in enumerate(enemies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, ngen + 1), avg_fitness_data[i], label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Average Fitness vs. Generation (Enemy {enemy})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, ngen + 1), max_fitness_data[i], label="Maximum Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Maximum Fitness vs. Generation (Enemy {enemy})")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Testing the best solutions
num_tests = 5
individual_gains = []

for enemy in enemies:
    env.update_parameter('enemies', [enemy])
    best_solutions = [best_of_gens[rep] for rep in range(nrep)]
    gains = []

    for solution in best_solutions:
        gains_rep = []

        for _ in range(num_tests):
            f, p, e, t = env.play(pcont=solution)
            individual_gain = p - e
            gains_rep.append(individual_gain)

        gains.append(gains_rep)

    individual_gains.append(gains)

# Plot box plots of individual gains for each enemy and algorithm
for i, enemy in enumerate(enemies):
    plt.figure(figsize=(8, 6))
    plt.boxplot(individual_gains[i], labels=[f"Algorithm {rep + 1}" for rep in range(nrep)])
    plt.xlabel("Algorithm")
    plt.ylabel("Individual Gain (Player Energy - Enemy Energy)")
    plt.title(f"Individual Gains Distribution (Enemy {enemy})")
    plt.show()

# Statistical analysis
for i, enemy in enumerate(enemies):
    gains = np.array(individual_gains[i])

    # Perform ANOVA test
    f_statistic, p_value = stats.f_oneway(*gains.T)
    print(f"ANOVA Test Results for Enemy {enemy}:")
    print(f"F-statistic: {f_statistic}")
    print(f"P-value: {p_value}")

# Save data to CSV
data = pd.DataFrame({'average fitness': avg_fitness_data, 'max fitness': max_fitness_data})
data.to_csv('fitness_data.csv')
