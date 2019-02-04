import sys
import argparse
import tqdm
from time import sleep

import ray
import ray.tune as tune


parser=argparse.ArgumentParser(
    description='''This script will execute a learning algorithms of our choice in a Ray environment, with the hyperparameters that we input.''')
parser.add_argument("--algo", type=str, help = "The algorithm of the model to train. Examples: DQN, DDQN, A2C, A3C")
parser.add_argument("--env", type=str, help = "OpenAI environment to train the model on.")
parser.add_argument("--stop", help = "Stop condition in which the execution will stop. Some samples of arguments to input: time_total_s: xxx, training_iteration: xxxx, episode_reward_mean: xxx. If not specified, the algorithm will train for 1 hour.", default = '{"time_total_s": 3600}')
parser.add_argument("--gpus", type=int, help = "Number of GPU units to be used during the training.", default=0)
parser.add_argument("--w", type=int, help = "Number of workers to be initialised during training. Each one will use one CPU core.", default = 1)
args = parser.parse_args()
print(args)


def __main__():
    
    #if len(args) != 5:
    #    raise ValueError("You must input the correct number of arguments. Use --help for more info.")

    if not ray.is_initialized():
        ray.init()
        
    #run_tune_algo()
    run_PPO()   
        
    
    

def run_tune_algo():
    tune.run_experiments({
        "my_experiment": {
            "run": args.algo,
            "env": args.env,
            "stop": args.stop,
            "config": {
                "num_gpus": args.gpus,
                "num_workers": args.w,
                #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
                "lr": 0.01,
            },
        },
    })
    
    
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