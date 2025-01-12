"""
Custom_Learning_v2

Se pretende hacer una implementación mas completa de las experimentaciones con RLLib, basándose en
los ejemplos que se encuentran en ray/python/ray/rllib/examples

Algoritmos posibles en RLLIB: PPO, DQN, A3C, A2C, IMPALA, PG (Policy Gradient), DDQN (comprobar), ...
Dado que las experimentaciones nos han dado problemas de memoria, probablemente lo mas seguro sea hacer
las ejecuciones con solo un worker, y un tamaño máximo de memoria a reservar (Probablemente entre 8-10 GB).

Además, vamos a probar a hacer una implementacion con una red neuronal propia con TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import math
import random
import numpy as np
#import tensorflow as tf
import gym
from gym.spaces import Box, Discrete, Dict

import ray
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.misc import normc_initializer
from ray.tune import run_experiments
#from ray.tune.registry import register_env

#from config import get_config

from aux_training import on_episode_start, on_episode_step, on_episode_end, on_train_result



parser=argparse.ArgumentParser(
    description='''This script will execute a learning algorithms of our choice in a Ray environment, with the hyperparameters that we input.''')
parser.add_argument("--run", type=str, help = "The algorithm of the model to train. Examples: DQN, DDQN, A2C, A3C", default = "DQN" )
parser.add_argument("--env", type=str, help = "OpenAI environment to train the model on.", default="BreakoutNoFrameskip-v4")
parser.add_argument("--stop", help = "Stop conditions Examples: time_total_s: xxx, training_iteration: xxxx, episode_reward_mean: xxx. If not specified, the algorithm will train for 1 hour.", default = { "training_iteration": int(3e6)})
parser.add_argument("--mem", help = "Maximum memory to be used, supposedly by a single agent process. Default: 2e9 (2 Gigabytes)", default=8e9)
parser.add_argument("--gpus", type=int, help = "GPUS to use. Default:1", default=1)
parser.add_argument("--w", type=int, help = "Number of workers", default = 1)
parser.add_argument("--folder", type=str, help = "Folder in which results will be stored. By default, they will end up in folder custom_experiments_v2.", default = "testing")
parser.add_argument("--all", type=bool, help = "If set to True it will execute all algorithms that we have as input, therefore the training time will be the stop condition times the number of algorithms that we execute.", default = False)

global default_config

def run_algo(algorithm):
    run_experiments({args.folder: {
            "run": algorithm,
            "env": args.env,
            "stop": args.stop,
            "config": get_config(algo)
        }})
    pass

def run_all_algos():

    algos = ["A2C", "A3C", "IMPALA", "DQN", "PPO", "ES"]


    for algo in algos:
        run_algo(algo)
    pass

if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    ray.init(object_store_memory=int(args.mem), num_gpus=args.gpus)
    if args.all:
        run_all_algos()
    else:
        run_algo(args.run)
