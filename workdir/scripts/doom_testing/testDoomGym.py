
import sys
import argparse
from time import sleep

##GYM

import gym
from gym import spaces

import vizdoomgym
from vizdoomgym.envs.vizdoomdefendcenter import VizdoomDefendCenter as env


##RAY
import ray
import ray.tune as tune

#from vizdoomgym.envs.vizdoomenv import VizdoomEnv as env # Settings of Doom environments

#from ray.rllib.models import Model, ModelCatalog
#from ray.tune.registry import register_env
from ray.tune import grid_search
from ray.tune import run_experiments



class DoomEnv(gym.Env):
    
    def __init__(self, config):
        super(DoomEnv, self).__init__()
        
        self.action_space = env.action_space # No lo está pillando
        self.observation_space = env.observation_space
        #self.shape = get_input_shape
        
    def reset(self):
        return env.reset()
        
    def step(self, action):
        return env.step(self, action)
        
    def render(self, mode):
        return env.render(self, mode)
        
    def get_keys_to_action():
        return env.get_keys_to_action()
        

def run_gym_test():
    env = gym.make('VizdoomDefendCenter-v0')
    state = env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
        
def run_single_algo():
    
    run_experiments({
        "testing": {
            "run": "PPO",
            "env": gym.make('VizdoomDefendCenter-v0'),
            "stop": {"time_total_s": 60},
            "config": dict({
                "num_workers": 1,
                #"sample_batch_size": 50,
                #"train_batch_size": 500,

            }),
        },
    })




if __name__ == "__main__":
    
    #args = parser.parse_args()
    #ray.init()
    #ModelCatalog.register_custom_model("myDoomEnv", DoomEnv) # Registers Doom Model (NOT ENVIRONMENT APPARENTLY)
    #run_single_algo()
    run_gym_test()
