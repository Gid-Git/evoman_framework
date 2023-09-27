import sys
import os
import time
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
import argparse
from deap import base, creator, tools, algorithms

# Define the DEAP fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

class EvolutionaryAlgorithm:
    def __init__(self, args):
        self.experiment_name = args.experiment_name
        self.n_hidden_neurons = args.n_hidden_neurons
        self.run_mode = args.run_mode
        self.gens = args.gens
        self.npop = args.npop
        self.mutation = args.mutation
        self.enemies = args.enemies
        self.headless = args.headless
        self.dom_u = 1
        self.dom_l = -1
        self.current_generation = 0
        self.total_generations = self.gens
        self.sigma_start = 3.0
        self.sigma_end = 0.1

        self.env = self.initialize_environment()
        self.n_vars = (self.env.get_num_sensors()+1)*self.n_hidden_neurons + (self.n_hidden_neurons+1)*5
        self.last_best = 0

    def initialize_population(self, n):
        return [creator.Individual(np.random.uniform(self.dom_l, self.dom_u, self.n_vars)) for _ in range(n)]

    def evaluate_individual(self, ind):
        f, p, e, t = self.env.play(pcont=ind)
        return f,

    def evaluate_population(self, population):
        return map(self.evaluate_individual, population),

    def mutate_individual(self, ind):
        sigma = self.get_sigma()
        ind += np.random.normal(0, sigma, len(ind))
        return ind,

    def run(self):
        if self.run_mode == 'train':
            self.train()
        elif self.run_mode == 'test':
            self.test()

    def train(self):
        # Create the DEAP toolbox
        toolbox = base.Toolbox()
        toolbox.register("population", self.initialize_population, self.npop)
        toolbox.register("individual", self.initialize_population, 1)
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Use Blend crossover
        toolbox.register("mutate", self.mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population()
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self.env.update_parameter('speed', 'normal')

        algorithms.eaMuPlusLambda(pop, toolbox, mu=self.npop, lambda_=self.npop, cxpb=0.7, mutpb=0.2, ngen=self.gens,
                                  stats=stats, halloffame=hof, verbose=True)

        best_individual = hof[0]
        self.save_results(best_individual)

        # saves results
        with open(self.experiment_name + '/results.txt', 'a') as file_aux:
            file_aux.write(f'\n{i} {round(self.fit_pop[best],6)} {round(mean,6)} {round(std,6)}')
            
        # saves generation number
        with open(self.experiment_name + '/gen.txt', 'w') as file_aux:
            file_aux.write(str(i))

        # saves file with the best solution
        np.savetxt(self.experiment_name + '/best.txt', self.pop[best])

        # saves simulation state
        solutions = [self.pop, self.fit_pop]
        self.env.update_solutions(solutions)
        self.env.save_state()

        self.current_generation += 1



def main():
    parser = argparse.ArgumentParser(description="Evolutionary Algorithm with EvoMan Framework.")
    parser.add_argument("--headless", action="store_true", help="Run without visualization for faster execution.")
    parser.add_argument("--experiment_name", default="optimization_test", help="Name of the experiment.")
    parser.add_argument("--run_mode", choices=["train", "test"], default="train", help="Mode to run the script: train or test.")
    parser.add_argument("--gens", type=int, default=30, help="Number of generations for the genetic algorithm.")
    parser.add_argument("--npop", type=int, default=100, help="Population size.")
    parser.add_argument("--mutation", type=float, default=0.2, help="Mutation rate.")
    parser.add_argument("--enemies", type=int, nargs='+', default=[8], help='List of enemies to fight')

    args = parser.parse_args()

    ea = EvolutionaryAlgorithm(args)
    ea.run()

    # Log the command
    if args.run_mode == "train":
        log_file_path = os.path.join(args.experiment_name, "commands_log.txt")
        with open(log_file_path, "a") as f:
            f.write(' '.join(sys.argv) + '\n')

if __name__ == "__main__":
    main()
