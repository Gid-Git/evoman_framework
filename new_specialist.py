import argparse
import numpy as np
from evoman.environment import Environment
from numpy.random import randint, randn

# Define a class for the genetic algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, tournament_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.env = Environment(speed="fastest")

    def initialize_population(self):
        return [np.random.randn(self.env.get_num_sensors(), self.env.get_num_actions()) for _ in range(self.population_size)]

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            individual += randn(*individual.shape)
        return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = randint(1, len(parent1))
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def tournament_selection(self, population, fitness):
        selected = np.random.choice(len(population), self.tournament_size)
        best = selected[np.argmax(fitness[selected])]
        return population[best]

    def train(self, generations):
        population = self.initialize_population()
        
        # Open the CSV file for writing
        csv_path = os.path.join(experiment_dir, 'results.csv')
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['Generation', 'Highest Fitness', 'Average Fitness', 'Standard Deviation Fitness', 'Health Gain'])
            
            for generation in range(generations):
                fitness = []
                for individual in population:
                    self.env.update_parameter('weights', individual)
                    _, player_energy, energy_enemy, _ = self.env.play(pcont=individual)
                    fitness.append(-energy_enemy)
                    
                # Calculate the required statistics
                highest_fitness = np.max(fitness)
                average_fitness = np.mean(fitness)
                std_dev_fitness = np.std(fitness)
                health_gain = player_energy - np.mean([e for e in self.env.get_enemys_energy()])
                
                # Write the statistics for the current generation to the CSV file
                writer.writerow([generation, highest_fitness, average_fitness, std_dev_fitness, health_gain])
                
                new_population = []
                for _ in range(self.population_size // 2):
                    parent1 = self.tournament_selection(population, fitness)
                    parent2 = self.tournament_selection(population, fitness)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(self.mutate(child1))
                    new_population.append(self.mutate(child2))
                population = new_population
            
        return self.tournament_selection(population, fitness)
    
# Define the argument parser
parser = argparse.ArgumentParser(description='Train a neural network using a genetic algorithm to defeat an enemy in EvoMan')
parser.add_argument("--experiment_name", default="specialist_test", help="Name of the experiment.")
parser.add_argument('--population_size', type=int, default=50, help='Size of the population')
parser.add_argument('--mutation_rate', type=float, default=0.01, help='Mutation rate')
parser.add_argument('--crossover_rate', type=float, default=0.7, help='Crossover rate')
parser.add_argument('--tournament_size', type=int, default=5, help='Tournament size for selection')
parser.add_argument('--generations', type=int, default=100, help='Number of generations')

args = parser.parse_args()

# Initialize and train the genetic algorithm
ga = GeneticAlgorithm(args.population_size, args.mutation_rate, args.crossover_rate, args.tournament_size)
best_weights = ga.train(args.generations)

print('Training complete. Best weights:', best_weights)
