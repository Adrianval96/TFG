# importing dependency libraries
from __future__ import print_function
import gym as Gym
import numpy as np
import time
import tqdm

#Load the environment

env = Gym.make('FrozenLake-v0')

s = env.reset()
print("initial state : ",s)
print()

env.render()
print()

print(env.action_space) #number of actions
print(env.observation_space) #number of states
print()

print("Number of actions : ",env.action_space.n)
print("Number of states : ",env.observation_space.n)
print()

#Epsilon-Greedy approach for Exploration and Exploitation of the state-action spaces
def epsilon_greedy(Q,s,na):
    epsilon = 0.1
    p = np.random.uniform(low=0,high=1)
    #print(p)
    if p > epsilon:
        return np.argmax(Q[s,:])#say here,initial policy = for each state consider the action having highest Q-value
    else:
        return env.action_space.sample()
		
# Q-Learning Implementation

#Initializing Q-table with zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

#set hyperparameters
alpha = 0.1 #learning rate
y = 0.6 #discount factor lambda
eps = 100000 #total episodes being 100000

for i in tqdm.tqdm(range(eps)):
#for i in range(eps):
    s = env.reset()
    t = False
    while(True):
        a = epsilon_greedy(Q,s,env.action_space.n)
        s_,r,t,_ = env.step(a) #next state, reward, done, info
        if (r==0): 
            if t==True:
                r = -5 #to give negative rewards when holes turn up
                Q[s_] = np.ones(env.action_space.n)*r #in terminal state Q value equals the reward
            else:
                r = -1 #to give negative rewards to avoid long routes
        if (r==1):
                r = 100
                Q[s_] = np.ones(env.action_space.n)*r #in terminal state Q value equals the reward
        Q[s,a] = (1-alpha) * Q[s,a] + alpha * (r + y * np.max(Q[s_]))
        #Q[s,a] = Q[s,a] + alpha * (r + y*np.max(Q[s_,a]) - Q[s,a])
        s = s_ 
        if (t == True) :
            break
    print(f"Episode: {i}")

print("Q-table")
print(Q)
print()

print("Output after learning")
print()
#learning ends with the end of the above loop of several episodes above
#let's check how much our agent has learned
s = env.reset()
env.render()
while(True):
    a = np.argmax(Q[s])
    s_,r,t,info = env.step(a)
    print("===============")
    print(f"Action: {a}, info: {info}")
    env.render()
    s = s_
    if(t==True) :
        break