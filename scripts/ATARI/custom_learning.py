import sys
import argparse
#import tqdm
from time import sleep

import ray
import ray.tune as tune


def __init__():
    
    #if len(args) != 5:
    #    raise ValueError("You must input the correct number of arguments. Use --help for more info.")   
    run_tune_algo(args.algo, args.env, args.stop, args.gpus, args.w)
    #run_PPO()   
        
    
#Ya no se utiliza
def run_tune_algo_old():
    var = {"my_experiment": {
        "run": args.algo,
        "env": args.env,
        "stop": args.stop,
        "config": {
            "num_gpus": args.gpus,
            "num_workers": args.w,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            #"lr": 0.01,
        },
    },
          }
    print(var)
    #tune.run_experiments(var) 
    
def run_single_algo(algo, env, stop, gpus, w):
    #TODO
    
    pass
    
def run_tune_algo(algo, env, stop, gpus, w):
    var = {"my_experiment": {
        "run": algo,
        "env": env,
        #"stop": stop,
        "stop": {"training_iteration": 600000, "time_total_s": 10800},
        "config": {
            "num_gpus": gpus,
            "num_workers": w,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            #"lr": 0.01,
        },
    },
          }
    print(var)
    tune.run_experiments(var) 
    
    
    
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
    
#Metodo para cuando meta algun parametro en plan testear todos los algoritmos
#La idea es ejecutar cada algoritmo en serie, o bien en paralelo cada uno con un worker, con el resto de parametros establecidos.
def algo_testing():
    pass
    
    
    
if __name__ == "__main__":
    __init__()
    
    
    parser=argparse.ArgumentParser(
        description='''This script will execute a learning algorithms of our choice in a Ray environment, with the hyperparameters that we input.''')
    parser.add_argument("--algo", type=str, help = "The algorithm of the model to train. Examples: DQN, DDQN, A2C, A3C")
    parser.add_argument("--env", type=str, help = "OpenAI environment to train the model on.")
    parser.add_argument("--stop", help = "Stop condition in which the execution will stop. Some samples of arguments to input: time_total_s: xxx, training_iteration: xxxx, episode_reward_mean: xxx. If not specified, the algorithm will train for 1 hour.", default = {"time_total_s": 3600})
    parser.add_argument("--gpus", type=int, help = "Number of GPU units to be used during the training.", default=0)
    parser.add_argument("--w", type=int, help = "Number of workers to be initialised during training. Each one will use one CPU core.", default = 1)
    args = parser.parse_args()
    print(args)
    
    
    if not ray.is_initialized():
        ray.init(object_store_memory=int(8e9))