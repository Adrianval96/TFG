# https://github.com/openai/wiki/MountainCar-v0




from IPython.display import clear_output
from time import sleep
import gym, random, numpy as np

#Hiperparámetros

alpha = 0.1 # Tasa de aprendizaje
gamma = 0.6 # Factor de descuento
epsilon = 0.1 # Factor de exploracion

# Contenedores de datos
all_epochs = []
all_penalties = []

env = gym.make("MountainCar-v0")

env.reset()

# Creo que esto no funciona porque este entorno no tiene unos estados explicitos
#q_table = np.zeros([env.observation_space, env.action_space])

#print(env.P) # {action: [(probability, nextstate, reward, done)]}, por alguna razón no funciona

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

#env.render()

epochs = 0 # timesteps
penalties, reward = 0, 0

frames = []

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

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
print("Penalties incurred: {}".format(penalties))
