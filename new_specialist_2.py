import numpy as np
import argparse
from evoman.environment import Environment
import os
from demo_controller import player_controller
import csv
import random

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

        # To keep track of all the health_gain_values
        self.health_gain_values = {}

        # Setup directories
        self.setup_directories()

        # Set up the Evoman environment
        self.env = self.initialize_environment()

        # Setup total network weights
        self.network_weights()
        
        # Initialize population
        if self.mode == 'train':
            self.population = [self.initialize_individual() for _ in range(self.population_size)]
        elif self.mode == "test":
            # Load the best individual from a file (adjust the filename as needed)
            self.best_individual = np.load('Results/experiment_1/best_individual.npy')

    def setup_directories(self):
        results_dir = os.path.join(script_dir, "Results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.experiment_dir = os.path.join(results_dir, self.experiment_name)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)



    # might not use! and only keep the n_inputs and outputs
    def network_weights(self):
        self.n_inputs = self.env.get_num_sensors()
        self.n_outputs = 5

        self.total_network_weights = ((self.n_inputs + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * self.n_outputs)

    def initialize_environment(self):
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        env = Environment(experiment_name=self.experiment_name,
                          enemies=self.enemy,
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=not self.headless)
        return env
    
    # def initialize_individual(self):
    #     # Initialize a neural network with random weights
    #     input_layer = np.random.randn(self.n_hidden_neurons, self.n_inputs + 1)
    #     hidden_layer = np.random.randn(self.n_outputs, self.n_hidden_neurons + 1)
    #     return (input_layer, hidden_layer)  
    
    def initialize_individual(self):
        # Initialize a neural network with random weights
        input_layer = np.random.randn(self.n_hidden_neurons, self.n_inputs + 1)
        hidden_layer = np.random.randn(self.n_outputs, self.n_hidden_neurons + 1)
        return np.concatenate((input_layer.flatten(), hidden_layer.flatten()))


    # def simulation(self, x):
    #     # if the health gain of this individual was already calculated return cached value
    #     individual_str = str(x)
    #     if individual_str in self.health_gain_values:
    #         return self.health_gain_values[individual_str]
        
    #     flattened_weights = np.concatenate((x[0].flatten(), x[1].flatten()))
        
    #     fitness, player_life, enemy_life, time = self.env.play(pcont=flattened_weights)
    #     health_gain = enemy_life - player_life

    #     self.health_gain_values[individual_str] = health_gain

    #     return health_gain

    def simulation(self, x):
        individual_str = str(x.tolist())
        if individual_str in self.health_gain_values:
            return self.health_gain_values[individual_str]
        
        fitness, player_life, enemy_life, time = self.env.play(pcont=x)
        health_gain = enemy_life - player_life

        self.health_gain_values[individual_str] = health_gain
        return health_gain

    
    def evaluate(self, x):
        # Evaluate individuals to get the life gain
        return np.array(list(map(lambda y: self.simulation(y), x)))
    
    # def mutate(self, individual):
    #     mutation_scale = 0.1  # Define a mutation scale

    #     # Apply Gaussian mutation to the individual
    #     for layer in individual:
    #         for i in range(layer.shape[0]):
    #             for j in range(layer.shape[1]):
    #                 if np.random.rand() < self.mutation_rate:
    #                     layer[i, j] += np.random.normal() # might add mutation scale
    #     return individual

    def mutate(self, individual):
        mutation_scale = 0.1  # Define a mutation scale
        
        # Reshape the flattened individual back to the original shape
        input_layer = individual[:self.n_hidden_neurons * (self.n_inputs + 1)].reshape(self.n_hidden_neurons, self.n_inputs + 1)
        hidden_layer = individual[self.n_hidden_neurons * (self.n_inputs + 1):].reshape(self.n_outputs, self.n_hidden_neurons + 1)
        
        # Apply Gaussian mutation to the individual layers
        for layer in (input_layer, hidden_layer):
            for i in range(layer.shape[0]):
                for j in range(layer.shape[1]):
                    if np.random.rand() < self.mutation_rate:
                        layer[i, j] += np.random.normal() * mutation_scale

        # Return the mutated individual as a flattened array
        return np.concatenate((input_layer.flatten(), hidden_layer.flatten()))

    
    # def crossover(self, parent1, parent2):
    #     # Perform one-point crossover on the parents
    #     child1 = (parent1[0].copy(), parent1[1].copy())
    #     child2 = (parent2[0].copy(), parent2[1].copy())
        
    #     for layer_index in range(2):
    #         crossover_point = np.random.randint(low=1, high=parent1[layer_index].shape[1])
    #         child1[layer_index][:, :crossover_point], child2[layer_index][:, :crossover_point] = \
    #         parent2[layer_index][:, :crossover_point], parent1[layer_index][:, :crossover_point]
        
    #     return child1, child2

    def crossover(self, parent1, parent2):
        # Reshape parents to original shapes
        parent1_input_layer = parent1[:self.n_hidden_neurons * (self.n_inputs + 1)].reshape(self.n_hidden_neurons, self.n_inputs + 1)
        parent1_hidden_layer = parent1[self.n_hidden_neurons * (self.n_inputs + 1):].reshape(self.n_outputs, self.n_hidden_neurons + 1)
        
        parent2_input_layer = parent2[:self.n_hidden_neurons * (self.n_inputs + 1)].reshape(self.n_hidden_neurons, self.n_inputs + 1)
        parent2_hidden_layer = parent2[self.n_hidden_neurons * (self.n_inputs + 1):].reshape(self.n_outputs, self.n_hidden_neurons + 1)

        # Perform one-point crossover on the parents
        child1_input_layer, child2_input_layer = parent1_input_layer.copy(), parent2_input_layer.copy()
        child1_hidden_layer, child2_hidden_layer = parent1_hidden_layer.copy(), parent2_hidden_layer.copy()
        
        crossover_point_input = np.random.randint(low=1, high=parent1_input_layer.shape[1])
        crossover_point_hidden = np.random.randint(low=1, high=parent1_hidden_layer.shape[1])
        
        child1_input_layer[:, :crossover_point_input], child2_input_layer[:, :crossover_point_input] = \
            parent2_input_layer[:, :crossover_point_input], parent1_input_layer[:, :crossover_point_input]

        child1_hidden_layer[:, :crossover_point_hidden], child2_hidden_layer[:, :crossover_point_hidden] = \
            parent2_hidden_layer[:, :crossover_point_hidden], parent1_hidden_layer[:, :crossover_point_hidden]

        # Return the children as flattened arrays
        return np.concatenate((child1_input_layer.flatten(), child1_hidden_layer.flatten())), \
            np.concatenate((child2_input_layer.flatten(), child2_hidden_layer.flatten()))

    
    def tournament_selection(self):
        # Select an individual using tournament selection
        tournament = random.sample(self.population, self.tournament_size)
        health_gains = [self.simulation(individual) for individual in tournament]
        return tournament[np.argmax(health_gains)]
    
    def train(self):
        if self.mode == 'train':
            # Filepaths
            self.csv_file_path = os.path.join(self.experiment_dir, 'results.csv')
            self.best_individual_path = os.path.join(self.experiment_dir, 'best_individual.npy')

            # Initialize CSV file with header
            with open(self.csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Generation", "Max Health Gain", "Min Health Gain", "Average Health Gain", "Stdv Health Gain"])

            # Run the evolutionary algorithm
            for generation in range(self.generations):
                print(f"Generation: {generation + 1}")

                # Clear the health_gains dictionary at the start of each generation
                self.health_gain_values.clear()
                
                # Select parents and create offspring
                offspring = []
                while len(offspring) < self.population_size:
                    parent1 = self.tournament_selection()
                    parent2 = self.tournament_selection()
                    if np.random.rand() < self.crossover_rate:
                        child1, child2 = self.crossover(parent1, parent2)
                        offspring.append(self.mutate(child1))
                        offspring.append(self.mutate(child2))
                    else:
                        offspring.append(self.mutate(parent1.copy()))
                        offspring.append(self.mutate(parent2.copy()))
                
                # Evaluate the offspring
                health_gains = self.evaluate(offspring)

                # Compute and write to csv data
                max_health_gain = np.max(health_gains)
                min_health_gain = np.min(health_gains)
                avg_health_gain = np.mean(health_gains)
                std_health_gain = np.std(health_gains)

                with open(self.csv_file_path, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([generation, max_health_gain, min_health_gain, avg_health_gain, std_health_gain])
                
                # Select the next generation
                self.population = [offspring[i] for i in np.argsort(health_gains)[-self.population_size:]]
                
                # Print the best health gain of the current generation
                print(f"Best health gain: {max_health_gain}")
            
                # Save the best individual to a file
                best_individual = self.population[np.argmax(health_gains)]
                np.save(self.best_individual_path, best_individual)
        
        elif self.mode == 'test':
            # Test the best individual
            self.headless = False
            self.env = self.initialize_environment()
            health_gain = self.evaluate([self.best_individual])
            print(f"Best individual health gain: {health_gain}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evoman Neural Network Training and Testing')
    parser.add_argument("--experiment_name", default="optimization_test", help="Name of the experiment.")
    parser.add_argument('--enemy', type=int, nargs='+', default=[8], help='List of enemies to fight.')
    parser.add_argument('--population_size', type=int, default=100, help='Size of the population')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.01, help='Mutation rate')
    parser.add_argument('--crossover_rate', type=float, default=0.7, help='Crossover rate')
    parser.add_argument('--tournament_size', type=int, default=5, help='Tournament size')
    parser.add_argument("--mode", type=str, choices=['train', 'test'], default='train', help='Training the EA or testing it.')
    parser.add_argument('--n_hidden_neurons', type=int, default=10, help='Number of neurons in the hidden layer')
    parser.add_argument("--headless", action="store_true", help="Run without visualization for faster execution.")
    
    args = parser.parse_args()
    
    # Create an instance of EvoManNN and run the algorithm
    evoman_n = EvoMan(args)
    evoman_n.train()
