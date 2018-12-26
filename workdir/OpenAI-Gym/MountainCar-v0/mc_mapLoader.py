import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import tqdm

import gym

env = gym.make('MountainCar-v0')



print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

state = env.reset()
