## Use Python3.6

"""training the agent"""

from IPython.display import clear_output
from time import sleep
import gym, random, numpy as np

env = gym.make("FrozenLake-v0")
env.reset()

epochs = 0
incompletes, reward = 0, 0

frames = []

done = False

for i in range (1, 1001):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if done and reward == 0:
        incompletes += 1

    if done and reward == 1:
        frames.append({
    		'frame': env.render(mode='ansi'),
    		'state': state,
    		'action': action,
    		'reward': reward
    		}
    	)

    epochs += 1

def print_frames(frames):
	for i, frame in enumerate(frames):
		clear_output(wait=True)
		print(frame['frame'].getvalue())
		print(f"Timestep: {i + 1}")
		print(f"State: {frame['state']}")
		print(f"Reward: {frame['reward']}")
		sleep(.25)

print_frames(frames)

print("Timesteps taken: {}".format(epochs))
print("Times fallen in hole: {}".format(incompletes))
