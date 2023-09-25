import numpy as np
import argparse
from evoman.environment import Environment
import os
from demo_controller import player_controller
import csv
import random
import sys

# Define the base file path
script_dir = os.path.dirname(os.path.abspath(__file__))

class EvoMan:
    def __init__(self, args):
        self.experiment_name = args.experiment_name
        self.enemy = args.enemy
        self.population_size = args.population_size
        self.generations = args.generations
        self.mutation_rate = args.mutation_rate
        self.crossover_rate = args.crossover_rate
        self.tournament_size = args.tournament_size
        self.mode = args.mode
        self.n_hidden_neurons = args.n_hidden_neurons
        self.headless = args.headless
        self.dom_l = args.dom_l
        self.dom_u = args.dom_u

        # Setup directories
        self.setup_directories()

        # Set up the Evoman environment
        self.env = self.initialize_environment()

        # Setup total network weights
        self.network_weights()

    def setup_directories(self):
        results_dir = os.path.join(script_dir, "Results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.experiment_dir = os.path.join(results_dir, self.experiment_name)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def network_weights(self):
        self.n_inputs = self.env.get_num_sensors()
        self.n_outputs = 5

        self.total_network_weights = ((self.n_inputs + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * self.n_outputs)

    def initialize_environment(self):
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        experiment_path = os.path.join("Results", self.experiment_name)

        env = Environment(experiment_name=experiment_path,
                          enemies=self.enemy,
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=not self.headless)
        return env
    
    def initialize_individual(self):
        # Initialize an individual with random weights and biases within the range [dom_l, dom_u]
        individual = np.random.uniform(self.dom_l, self.dom_u, self.total_network_weights)
        return individual
    
    def simulation(self, x):
        fitness, player_life, enemy_life, time = self.env.play(pcont=x)
        health_gain = enemy_life - player_life

        return fitness, health_gain, time
    
    def evaluate(self, population):
        # Evaluates the entire population and returns the fitness values
        fitness = np.zeros(self.population_size)
        health_gain = np.zeros(self.population_size)
        time = np.zeros(self.population_size)
        for i, individual in enumerate(population):
            fitness[i], health_gain[i], time[i] = self.simulation(individual)
        return fitness, health_gain, time
    
    def mutate(self, individual):
        # Applies mutation to the individual based on the mutation rate
        for i in range(len(individual)):
            if random.uniform(0, 1) < self.mutation_rate:
                individual[i] = np.random.uniform(self.dom_l, self.dom_u)
        return individual
    
    def crossover(self, parent1, parent2):
        # Applies crossover based on the crossover rate and returns the offspring
        if random.uniform(0, 1) < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
        
    def tournament_selection(self, population, fitness):
        # Selects an individual using tournament selection and returns the selected individual
        selected_indices = np.random.choice(self.population_size, self.tournament_size)
        tournament_individuals = population[selected_indices]
        tournament_fitness = fitness[selected_indices]
        winner_index = np.argmax(tournament_fitness)
        return tournament_individuals[winner_index]
    
    def train(self):
        # Initialize population
        population = np.array([self.initialize_individual() for _ in range(self.population_size)])

        # Evaluate the initial population
        fitness, health_gain, time = self.evaluate(population)

        # Initialize best individual and its fitness
        best_individual_index = np.argmax(fitness)
        best_individual = population[best_individual_index]
        best_fitness = fitness[best_individual_index]

        # Creat csv file
        results_file_path = os.path.join(self.experiment_dir, "results.csv")
        with open(results_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Best Fitness", "Average Fitness", "Std Fitness", "Best Health",
                             "Avg Health", "Std Health", "Lowest Time", "Avg Time", "Std Time"])
            # Save initial population
            writer.writerow([0, np.max(fitness), np.mean(fitness), np.std(fitness),
                    np.max(health_gain), np.mean(health_gain), np.std(health_gain),
                    np.min(time), np.mean(time), np.std(time)])
            # Main loop for generations
            for gen in range(self.generations):
                new_population = []
                for _ in range(self.population_size // 2):  # Two children per iteration
                    parent1 = self.tournament_selection(population, fitness)
                    parent2 = self.tournament_selection(population, fitness)

                    child1, child2 = self.crossover(parent1, parent2)

                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    new_population.extend([child1, child2])

                population = np.array(new_population)
                fitness, health_gain, time = self.evaluate(population)

                # Check if any individual has a higher fitness, save that one
                max_fitness_index = np.argmax(fitness)
                if fitness[max_fitness_index] > best_fitness:
                    best_fitness = fitness[max_fitness_index]
                    best_individual = population[max_fitness_index]

                # save to csv
                writer.writerow([gen, np.max(fitness), np.mean(fitness), np.std(fitness),
                                    np.max(health_gain), np.mean(health_gain), np.std(health_gain),
                                    np.min(time), np.mean(time), np.std(time)])
                
                print(f"Generation {gen}, Best Fitness: {np.max(fitness)}, best health: {np.max(health_gain)}, best time: {np.min(time)}")
                
        # Save the best individual's neural network weights
        np.save(os.path.join(self.experiment_dir, "best_individual.npy"), best_individual)

    def main():
        parser = argparse.ArgumentParser(description="Evolutionary Algorithm for EvoMan")
        
        parser.add_argument("--experiment_name", type=str, default="experiment", help="Name of the experiment")
        parser.add_argument("--enemy", type=int, default=[4], help="Enemy number")
        parser.add_argument("--population_size", type=int, default=100, help="Size of the population")
        parser.add_argument("--generations", type=int, default=50, help="Number of generations")
        parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
        parser.add_argument("--crossover_rate", type=float, default=0.9, help="Crossover rate")
        parser.add_argument("--tournament_size", type=int, default=5, help="Tournament size for selection")
        parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
        parser.add_argument("--n_hidden_neurons", type=int, default=10, help="Number of hidden neurons")
        parser.add_argument("--headless", action="store_true", help="Run in headless mode")
        parser.add_argument("--dom_l", type=float, default=-1, help="Lower bound for initialization and mutation")
        parser.add_argument("--dom_u", type=float, default=1, help="Upper bound for initialization and mutation")

        args = parser.parse_args()

        evoman = EvoMan(args)
        evoman.train()

        # Log the command
        if args.mode == "train":
            log_file_path = os.path.join(evoman.experiment_dir, "commands_log.txt")
            with open(log_file_path, "a") as f:
                f.write(' '.join(sys.argv) + '\n')

if __name__ == "__main__":
    EvoMan.main()