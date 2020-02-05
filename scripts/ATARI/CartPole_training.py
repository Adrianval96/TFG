import sys
import argparse
import tqdm
from time import sleep

import ray
import ray.tune as tune


def __main__():
    
    #if len(args) != 5:
    #    raise ValueError("You must input the correct number of arguments. Use --help for more info.")

    if not ray.is_initialized():
        ray.init()
        
    #run_tune_algo()  
    run_A3C()

    def run_PPO():
        tune.run_experiments({
            "my_experiment": {
                "run": "PPO",
                "env": "CartPole-v0",
                "stop": {"episode_reward_mean": 200},
                "config": {
                    "num_gpus": 1,
                    "num_workers": 4,
                    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
                },
            },
        })
        
        run_PPO() 

def run_A3C():
    tune.run_experiments({
    "my_experiment": {
        "run": "A3C",
        "env": "CartPole-v0",
        "stop": {"episode_reward_mean": 200},
        "config": {
            "num_gpus": 1,
            "num_workers": 4,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        },
    },
})
    
if __name__ == "__main__":
    __main__()