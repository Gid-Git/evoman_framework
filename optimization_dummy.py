###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


    dom_u = 1
    dom_l = -1
    npop = 100
    gens = 30
    mutation = 0.2
    last_best = 0

    k = 4
    # tournament
    def tournament(pop, fit_pop, k):
        c1 =  np.random.randint(0,pop.shape[0], 1)
        for i in range(k):
            c2 =  np.random.randint(0,pop.shape[0], 1)
            if fit_pop[c2] > fit_pop[c1]:
                c1 = c2
        return pop[c1][0]


    # start writing your own code from here
    def selection(pop):

        total_offspring = np.zeros((0,n_vars))


        for p in range(0,pop.shape[0], 2):
            p1 = tournament(pop)
            p2 = tournament(pop)

            n_offspring =   np.random.randint(1,3+1, 1)[0]
            offspring =  np.zeros( (n_offspring, n_vars) )

            for f in range(0,n_offspring):

                cross_prop = np.random.uniform(0,1)
                offspring[f] = p1*cross_prop+p2*(1-cross_prop)

                # mutation
                for i in range(0,len(offspring[f])):
                    if np.random.uniform(0 ,1)<=mutation:
                        offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

                offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring



if __name__ == '__main__':
    main()