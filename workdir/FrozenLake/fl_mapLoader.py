import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import tqdm

import gym

from ai_frozen import Dqn

env = gym.make('FrozenLake-v0')

print("Initial state: ", env.reset())
print("-----------------------------")

env.render()
print("-----------------------------")

nb_action = env.action_space.n

brain = Dqn(1, nb_action, 0.9)

eps =  100000

class Game():
    def __init__():
        last_reward = 0
        last_state = env.reset()
        scores = []

    def perform_action(a):
        return env.step(a)
    ###########################################################################
    # Q-Learning method that might be replaced once the Neural Network is fully implemented
    def epsilon_greedy(Q,s,na):
        epsilon = 0.1
        p = np.random.uniform(low=0,high=1)
        #print(p)
        if p > epsilon:
            return np.argmax(Q[s,:])#say here,initial policy = for each state consider the action having highest Q-value
        else:
            return env.action_space.sample()
    ##########################################################################

    def update(self, dt):

        global brain
        global last_reward
        global last_state
        global scores

        #action = epsilon_greedy(Q,s,env.action_space.n)
        ## This next line must be updated so the action will be decided by the NN
        #last_state, last_reward, done, _ = env.step(action)

        action, new_reward = brain.update(last_state)
        scores.append(brain.score())
