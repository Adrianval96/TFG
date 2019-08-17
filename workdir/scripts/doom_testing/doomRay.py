import gym
from gym import spaces

##RAY
import ray
import ray.tune as tune

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

    ray.init()
    ModelCatalog.register_custom_model("myDoomEnv", DoomEnv) # Registers Doom Model (NOT ENVIRONMENT APPARENTLY)
    run_single_algo()
