## Use Python3.6

from IPython.display import clear_output
from time import sleep

import gym

## R,G,Y,B: Possible pickup/destination locations

env = gym.make("Taxi-v2").env

env.reset()

env.s = 328

epochs = 0 #Timesteps
penalties, reward = 0, 0

frames = [] #for animation

done = False

while not done:
	action = env.action_space.sample()
	state, reward, done, info = env.step(action)
	
	if reward == -10:
		penalties += 1
		
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
		sleep(.1)

print_frames(frames)

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


		
