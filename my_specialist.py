import sys
import os
import time
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
import argparse


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
        self.sigma_start = args.sigma_start
        self.sigma_end = args.sigma_end
        self.n_elitism = args.n_elitism

        self.    = self.initialize_environment()
        self.n_vars = (self.env.get_num_sensors()+1)*self.n_hidden_neurons + (self.n_hidden_neurons+1)*5
        self.last_best = 0

    def initialize_environment(self):
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        env = Environment(experiment_name=self.experiment_name,
                          enemies=self.enemies,
                          playermode="ai",
                          player_controller=player_controller(self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=not self.headless)
        return env

    def simulation(self, x):
        f, p, e, t = self.env.play(pcont=x)
        return f

    def norm(self, x, pfit_pop):
        if (max(pfit_pop) - min(pfit_pop)) > 0:
            x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
        else:
            x_norm = 0

        return max(x_norm, 0.0000000001)

    def evaluate(self, x):
        return np.array(list(map(lambda y: self.simulation(y), x)))

    def tournament(self, pop):
        c1 = np.random.randint(0, pop.shape[0], 1)
        c2 = np.random.randint(0, pop.shape[0], 1)

        return pop[c1][0] if self.fit_pop[c1] > self.fit_pop[c2] else pop[c2][0]

    def limits(self, x):
        return max(min(x, self.dom_u), self.dom_l)

    def crossover(self):
        total_offspring = np.zeros((0, self.n_vars))

        for p in range(0, self.pop.shape[0], 2):
            p1 = self.tournament(self.pop)
            p2 = self.tournament(self.pop)

            n_offspring = np.random.randint(1, 3 + 1, 1)[0]
            offspring = np.zeros((n_offspring, self.n_vars))

            for f in range(n_offspring):
                for i in range(self.n_vars):
                    # Uniform crossover
                    if np.random.uniform(0, 1) < 0.5:
                        offspring[f][i] = p1[i]
                    else:
                        offspring[f][i] = p2[i]

                    # Non-uniform Gaussian mutation, with sigma
                    if np.random.uniform(0, 1) <= self.mutation:
                        sigma = self.get_sigma()
                        offspring[f][i] += np.random.normal(0, sigma)

                offspring[f] = np.array(list(map(self.limits, offspring[f])))
                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring
    
    def get_sigma(self):
        if self.current_generation < self.total_generations:
            sigma = self.sigma_start - self.current_generation * (self.sigma_start - self.sigma_end) / self.total_generations
            return max(sigma, self.sigma_end)
        else:
            return self.sigma_end

    def doomsday(self, pop, fit_pop):
        worst = int(self.npop / 4)
        order = np.argsort(fit_pop)
        orderasc = order[:worst]

        for o in orderasc:
            for j in range(self.n_vars):
                pro = np.random.uniform(0, 1)
                if np.random.uniform(0, 1) <= pro:
                    pop[o][j] = np.random.uniform(self.dom_l, self.dom_u)
                else:
                    pop[o][j] = pop[order[-1:]][0][j]

            fit_pop[o] = self.evaluate([pop[o]])

        return pop, fit_pop
    
    def test(self):
        bsol = np.loadtxt(self.experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        self.env.update_parameter('speed', 'normal')
        self.evaluate([bsol])

    def run(self):
        if self.run_mode == 'train':
            self.train()
        elif self.run_mode == 'test':
            self.test()

    def elitism(self, k):
        """
        Select the top k individuals
        """
        best_indices = np.argsort(self.fit_pop)[-k:]
        return self.pop[best_indices], self.fit_pop[best_indices]

    def train(self):

        if not os.path.exists(self.experiment_name + '/evoman_solstate'):
            print('\nNEW EVOLUTION\n')
            self.pop = np.random.uniform(self.dom_l, self.dom_u, (self.npop, self.n_vars))
            self.fit_pop = self.evaluate(self.pop)
            best = np.argmax(self.fit_pop)
            ini_g = 0
            solutions = [self.pop, self.fit_pop]
            self.env.update_solutions(solutions)
        else:
            print('\nCONTINUING EVOLUTION\n')
            self.env.load_state()
            self.pop = self.env.solutions[0]
            self.fit_pop = self.env.solutions[1]
            best = np.argmax(self.fit_pop)
            # finds last generation number
            file_aux = open(self.experiment_name + '/gen.txt', 'r')
            ini_g = int(file_aux.readline())
            file_aux.close()

        last_sol = self.fit_pop[best]
        notimproved = 0

        for i in range(ini_g + 1, self.gens):
            offspring = self.crossover()  # crossover
            fit_offspring = self.evaluate(offspring)   # evaluation
            self.pop = np.vstack((self.pop, offspring))
            self.fit_pop = np.append(self.fit_pop, fit_offspring)

            best = np.argmax(self.fit_pop)
            self.fit_pop[best] = float(self.evaluate(np.array([self.pop[best]]))[0])
            best_sol = self.fit_pop[best]

            # selection with elitism
            fit_pop_cp = self.fit_pop
            fit_pop_norm = np.array(list(map(lambda y: self.norm(y, fit_pop_cp), self.fit_pop)))
            probs = (fit_pop_norm) / (fit_pop_norm).sum()
            chosen = np.random.choice(self.pop.shape[0], self.npop-2, p=probs, replace=False)
            elite_pop, elite_fit = self.elitism(self.n_elitism)
            self.pop = np.vstack((self.pop[chosen], elite_pop))
            self.fit_pop = np.append(self.fit_pop[chosen], elite_fit)

            if best_sol <= last_sol:
                notimproved += 1
            else:
                last_sol = best_sol
                notimproved = 0

            if notimproved >= 15:
                self.pop, self.fit_pop = self.doomsday(self.pop, self.fit_pop)
                notimproved = 0

            best = np.argmax(self.fit_pop)
            std = np.std(self.fit_pop)
            mean = np.mean(self.fit_pop)

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
    parser.add_argument("--n_hidden_neurons", type=int, default=10, help="Number of hidden neurons.")
    parser.add_argument("--run_mode", choices=["train", "test"], default="train", help="Mode to run the script: train or test.")
    parser.add_argument("--gens", type=int, default=30, help="Number of generations for the genetic algorithm.")
    parser.add_argument("--npop", type=int, default=100, help="Population size.")
    parser.add_argument("--mutation", type=float, default=0.2, help="Mutation rate.")
    parser.add_argument("--enemies", type=int, nargs='+', default=[8], help='List of enemies to fight.')
    parser.add_argument("--n_elitism", type=int, default=2, help="Number of best individuals from population that are always selected for the next generation.")
    parser.add_argument("--sigma_start", type=float, default=1.0, help="Starting sigma for mutation strenght.")
    parser.add_argument("--sigma_end", type=int, default=0.1, help="Ending sigma for mutation strength.")

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