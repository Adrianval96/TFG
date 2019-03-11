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

parser=argparse.ArgumentParser(
    description='''This script will execute a learning algorithms of our choice in a Ray environment, with the hyperparameters that we input.''')
parser.add_argument("--run", type=str, help = "The algorithm of the model to train. Examples: DQN, DDQN, A2C, A3C", default = "DQN" )
parser.add_argument("--env", type=str, help = "OpenAI environment to train the model on.", default="CartPole-v0")
parser.add_argument("--stop", help = "Stop conditions Examples: time_total_s: xxx, training_iteration: xxxx, episode_reward_mean: xxx. If not specified, the algorithm will train for 1 hour.", default = {"time_total_s": 7200})
parser.add_argument("--gpus", type=int, help = "GPUS to use. Default:1", default=1)
parser.add_argument("--w", type=int, help = "Number of workers", default = 1)

def run_all_algos():
    
    algos = ["A3C", "A2C", "DQN", "IMPALA", "PPO"]
    
    for algo in algos:
        run_experiments({"single_run": {
            "run": algo,
            "env": args.env,
            "stop": {"training_iteration": 600000, "time_total_s": 7200}, #Default: 2 horas (7200 s)
            "config": dict({
                "num_gpus": args.gpus,
                "num_workers": args.w,
                "sample_batch_size": 50,
                "train_batch_size": 500,
                #"lr_schedule": [0, 0.0005],
                })
            }
        })
    
    pass
  
def run_single_algo():
    
    run_experiments({
        "test_run": {
            "run": args.run,
            "env": args.env,
            "stop": {"training_iteration": 600000, "time_total_s": 10800},
            "config": dict({
                "num_gpus": args.gpus,
                "num_workers": args.w,
                "sample_batch_size": 50,
                "train_batch_size": 500,
                #"lr_schedule": [0, 0.0005],

            }),
        },
    })
    
if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    
    ray.init(object_store_memory=int(8e9))

    run_single_algo()

    #__init__()
