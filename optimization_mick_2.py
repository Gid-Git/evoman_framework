import math
import numpy as np
import argparse
from evoman.environment import Environment
import os
from demo_controller import player_controller
import csv
import random
import sys
import time

# Define the base file path
script_dir = os.path.dirname(os.path.abspath(__file__))

class EvoMan:
    def __init__(self, experiment_name, enemy, population_size, generations, mutation_rate, crossover_rate, 
                 tournament_size, mode, n_hidden_neurons, headless, dom_l, dom_u, speed, n_elitism, k_tournament, sel_pres_incr, k_tournament_final):
        self.experiment_name = experiment_name
        self.enemy = enemy
        self.n_pop = population_size
        self.gens = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.mode = mode
        self.n_hidden_neurons = n_hidden_neurons
        self.headless = headless
        self.dom_l = dom_l
        self.dom_u = dom_u
        self.speed = speed
        self.n_elitism = n_elitism
        self.k_tournament = k_tournament
        self.sel_pres_incr = sel_pres_incr
        self.k_tournament_final = k_tournament_final
        self.current_generation = 0

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
                          speed=self.speed,
                          visuals=not self.headless)
        return env
    
    def initialize_individual(self):
        # Initialize an individual with random weights and biases within the range [dom_l, dom_u]
        individual = np.random.uniform(self.dom_l, self.dom_u, self.total_network_weights)
        return individual
    
    def simulation(self, x):
        fitness, player_life, enemy_life, time = self.env.play(pcont=x)
        health_gain = player_life - enemy_life

        return fitness, health_gain, time
    
    def evaluate(self, population):
        # Evaluates the entire population and returns the fitness values
        fitness = np.zeros(self.n_pop)
        health_gain = np.zeros(self.n_pop)
        time_game = np.zeros(self.n_pop)
        for i, individual in enumerate(population):
            fitness[i], health_gain[i], time_game[i] = self.simulation(individual)
        return fitness, health_gain, time_game

    def mutate(self, individual):
        # Applies mutation to the individual based on the mutation rate
        for i in range(len(individual)):
            if random.uniform(0, 1) < self.mutation_rate:
                mutation = np.random.normal(individual[i], 0.1)
                individual[i] = np.clip(mutation, self.dom_l, self.dom_u)
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
    
    # tournament (returns winnning individual and its fitness)
    def tournament_selection(self, population, fitness, k=2):
        # Generate k unique random indices
        random_indices = np.random.choice(self.n_pop, k, replace=False)
        # Select the individuals with the highest fitness values among the randomly chosenpytrel ones
        best_individual_index = np.argmax(fitness[random_indices])
        # Return the index of the best individual on the population scale
        winner_index = random_indices[best_individual_index]

        return population[winner_index], fitness[winner_index], winner_index
    
    def elitism(self, k, population, fitness):
        """
        Select the top k individuals
        """
        best_indices = np.argsort(fitness)[-k:]

        elite_pop = population[best_indices]
        elite_fit = fitness[best_indices]

        # Remove the selected elite individuals and their fitness values from population and fitness
        population = np.delete(population, best_indices, axis=0)
        fitness = np.delete(fitness, best_indices)

        return elite_pop, elite_fit, population, fitness, best_indices


    # returns selected individuals and their fitness
    def selection(self, population, fitness):
        selected = []
        selected_fit = []
        #Now elitism function also deletes the elites from self.pop and self.pop_fit so they cant be chosen in the selection step
        elite_pop, elite_fit, population, fitness, best_indices = self.elitism(self.n_elitism, population, fitness)

        #if selection pressure is set to True we want the pressure to increase and thus the number of neural networks to compete to increase
        #eventually the number of individuals in tournament is doubled from start to finish.
        if self.sel_pres_incr:
            k = max(math.ceil(self.current_generation*self.k_tournament_final/self.gens)*self.k_tournament, self.k_tournament)
        else:
            k = self.k_tournament
        for p in range(self.n_pop-self.n_elitism):
            select, fit, winner_index = self.tournament_selection(population, fitness, k)
            # Remove the selected elite individuals and their fitness values from population and fitness
            population = np.delete(population, winner_index, axis=0)
            fitness = np.delete(fitness, winner_index)
            selected.append(select)
            selected_fit.append(fit)
            best_indices = np.append(best_indices, winner_index)

        # print(f'fitness: {fitness.shape}, type: {type(fitness)}')
        # print(f'population: {population.shape}, type: {type(population)}')
        # Add the elite individuals to selected
        population = np.concatenate((selected, elite_pop), axis=0)
        fitness = np.concatenate((selected_fit, elite_fit), axis=0)
        return population, fitness, best_indices
    
    def run(self):
        if self.mode == "train":
            self.train()
        elif self.mode == "test":
            self.test()
    
    def train(self):
        # take time of entire run
        start_time = time.time()

        # Initialize population
        population = np.array([self.initialize_individual() for _ in range(self.n_pop)])

        # Evaluate the initial population
        fitness, health_gain, time_game = self.evaluate(population)

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
                    np.min(time_game), np.mean(time_game), np.std(time_game)])

            # Main loop for generations
            for gen in range(self.gens):
                children = []
                self.current_generation += 1
                for _ in range(self.n_pop // 2):  # Two children per iteration
                    parent1, fitness1, winn = self.tournament_selection(population, fitness)
                    parent2, fitness1, winn = self.tournament_selection(population, fitness)

                    child1, child2 = self.crossover(parent1, parent2)

                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    children.extend([child1, child2])    
                fitness_children, health_gain_children, time_game_children = self.evaluate(children)
                health_gain_before_selection = np.concatenate((health_gain, health_gain_children))
                time_game_before_selection = np.concatenate((time_game, time_game_children))
                pop_before_selection = np.concatenate((population, children))
                fitness_before_selection = np.concatenate((fitness, fitness_children))
                population, fitness, selected_indices = self.selection(pop_before_selection, fitness_before_selection)
                health_gain = health_gain_before_selection[selected_indices]
                time_game = time_game_before_selection[selected_indices]

                # Check if any individual has a higher fitness, save that one
                max_fitness_index = np.argmax(fitness)
                print(max_fitness_index)
                if fitness[max_fitness_index] > best_fitness:
                    best_fitness = fitness[max_fitness_index]
                    best_individual = population[max_fitness_index]

                # save to csv
                writer.writerow([gen, np.max(fitness), np.mean(fitness), np.std(fitness),
                                    np.max(health_gain), np.mean(health_gain), np.std(health_gain),
                                    np.min(time_game), np.mean(time_game), np.std(time_game)])
                
                print(f"Generation {gen}, Best Fitness: {np.max(fitness)}")
                
                # Save the best individual's neural network weights
                np.save(os.path.join(self.experiment_dir, "best_individual.npy"), best_individual)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save the elapsed time
        time_file_path = os.path.join(self.experiment_dir, "training_time.txt")
        with open(time_file_path, 'w') as file:
            file.write(f"Training Time: {elapsed_time} seconds\n")

    def test(self):
        best_individual_path = os.path.join(self.experiment_dir, "best_individual.npy")

        if os.path.exists(best_individual_path):
            # Load the best individual's neural network weights
            best_individual = np.load(best_individual_path)

            # Run the simulation with the best individual
            fitness, health_gain, time = self.simulation(best_individual)

        else:
            print("No best individual found!")


def run_evoman(experiment_name, enemy, population_size, generations, mutation_rate, crossover_rate, tournament_size, mode, 
               n_hidden_neurons, headless, dom_l, dom_u, speed, n_elitism, k_tournament, sel_pres_incr, k_tournament_final):
        evoman = EvoMan(experiment_name, enemy, population_size, generations, mutation_rate, crossover_rate, 
                        tournament_size, mode, n_hidden_neurons, headless, dom_l, dom_u, speed, n_elitism, k_tournament, sel_pres_incr, k_tournament_final)
        
        # Log the command
        if mode == "train":
            log_file_path = os.path.join(evoman.experiment_dir, "commands_log.txt")
            with open(log_file_path, "a") as f:
                f.write(' '.join(sys.argv) + '\n')
        
        evoman.run()
        

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Evolutionary Algorithm for EvoMan")
        
        parser.add_argument("--experiment_name", type=str, default="experiment_enemy=6_k_increase=3", help="Name of the experiment")
        parser.add_argument("--enemy", type=int, nargs='+',default=[6], help="Enemy number")
        parser.add_argument("--npop", type=int, default=100, help="Size of the population")
        parser.add_argument("--gens", type=int, default=50, help="Number of generations")
        parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
        parser.add_argument("--crossover_rate", type=float, default=0.9, help="Crossover rate")
        parser.add_argument("--tournament_size", type=int, default=5, help="Tournament size for selection")
        parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
        parser.add_argument("--n_hidden_neurons", type=int, default=10, help="Number of hidden neurons")
        parser.add_argument("--headless", action="store_true", help="Run in headless mode")
        parser.add_argument("--dom_l", type=float, default=-1, help="Lower bound for initialization and mutation")
        parser.add_argument("--dom_u", type=float, default=1, help="Upper bound for initialization and mutation")
        parser.add_argument("--speed", type=str, default="fastest", help="Speed: fastest or normal")
        parser.add_argument("--n_elitism", type=int, default=2, help="Number of best individuals from population that are always selected for the next generation.")
        parser.add_argument("--k_tournament", type=int, default= 3, help="The amount of individuals to do a tournament with for selection, the more the higher the selection pressure")
        parser.add_argument("--selection_pressure_increase", type=bool, default=True, help="if set to true the selection pressure will linearly increase over time from k_tournament till 2*k_tournament")
        parser.add_argument("--k_tournament_final", type=int, default= 3, help="The factor with which k_tournament should linearly increase (if selection_pressure_increase = True), if the value is 4 the last quarter of generations have tournaments of size k_tournament*4")

        args = parser.parse_args()

        run_evoman(args.experiment_name, args.enemy, args.npop, args.gens, args.mutation_rate, args.crossover_rate,
               args.tournament_size, args.mode, args.n_hidden_neurons, args.headless, args.dom_l, args.dom_u, args.speed, 
               args.n_elitism, args.k_tournament, args.selection_pressure_increase, args.k_tournament_final)