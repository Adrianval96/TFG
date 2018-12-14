import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import tqdm

import gym

env = Gym.make('Frozenlake-v0')

s = env.reset()
print("Initial state: ", s)
print("-----------------------------")

env.render()
print("-----------------------------")

nb_action = env.action_space.n

brain = Dqn(1, nb_action, 0.9)
last_reward = 0
scores = []

last_state = 0
eps =  100000

class Game():

    def update(self, dt):

        global brain
        global last_reward
        global scores
