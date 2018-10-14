## Use Python3.6

"""training the agent"""

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

env = gym.make("Taxi-v2")

# Q-Table [Acciones x Estados]
q_table = np.zeros([env.observation_space.n, env.action_space.n]) 

for i in range (1, 100001):
	state = env.reset()
	
	epochs, penalties, reward = 0, 0, 0
	done = False
	
	while not done:
		if random.uniform(0,1) < epsilon:
			action = env.action_space.sample() # Acción aleatoria a explorar
		else: 
			action = np.argmax(q_table[state]) # Comprueba la Q-Table y elige la mejor acción
			
		next_state, reward, done, info = env.step(action)
		
		old_value = q_table[state, action]
		next_max = np.max(q_table[next_state]) # Recompensa maxima del proximo estado
		
		new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # Función de aprendizaje
		q_table[state, action] = new_value
		
		if reward == -10:
			penalties += 1
			
		state = next_state
		epochs += 1
		
	if i % 100 == 0:
		clear_output(wait=True)
		print(f"Episode: {i}")
		
print("Training finished.\n")

def test_performance():

	print("------------Evaluating performance--------------\n")

	total_epochs, total_penalties = 0, 0
	episodes = 100

	for _ in range(episodes):
		state = env.reset()
		epochs, penalties, reward = 0, 0, 0
		
		done = False
		
		while not done:
			action = np.argmax(q_table[state])
			state, reward, done, info = env.step(action)
			
			if reward == -10:
				penalties +=1 
				
			epochs += 1
			
		total_penalties += penalties
		total_epochs += epochs
		
	print(f"Results after {episodes} episodes:")
	print(f"Average timesteps per episode: {total_epochs / episodes}")
	print(f"Average penalties per episode: {total_penalties / episodes}")

test_performance()
