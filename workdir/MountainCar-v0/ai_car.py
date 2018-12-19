# Inspired in code from AI from A-Z

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

class Dqn():

	# Create and initialize all needed variables
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma	# Gamma = Discount factor
        self.reward_window = []
        self.model = Network(input_size, nb_action) # Creates the object of the network class
        self.memory = ReplayMemory(100000) # Object of memory with capacity = 100000
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.01) # Creates object from torch.optim as optimizer with the parameters from the model and learning rate = 0.001
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # Creates a batch to input in the NN with the variables of the state and a fake dimension in position 0 corresponding to the batch
        self.last_action = 0 # {0,1}
        self.last_reward = 0 # [-1,0,+1]

    # This probably needs to be changed so it selects an action from the possible pool
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100 (temperature parameter)
        action = probs.multinomial(num_samples=1) # Returns a random action from the possible actions based on a multinomial distribution
        return action.data[0,0] # The action is contained in index [0,0] of the data

	# This method implements Deep Q-Learning process represented in Handbook - Chapter 5.3
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # (dqn object, current state, next state, reward, performed action) ## MARKOV DECISION PROCESS
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #gather returns the best action to play for each of the input states
																						  #batch_action.unsqueeze(1) to make it consistent with batch_state
																						  #.squeeze(1) kills the fake dimension since we don't need the batch anymore outside the neural network
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward	# Target Q-Value.
        td_loss = F.smooth_l1_loss(outputs, target) # Inputs predictions and targets to loss function to compare results
        self.optimizer.zero_grad() # Reinitializes optimizer to a gradient = 0
        td_loss.backward() # Backpropagation
        self.optimizer.step() # Updates the weights of the NN

    def update(self, state):
        new_state = torch.Tensor(state).float().unsqueeze(0) # Converts the signals of our three sensors into a Torch tensor with floating comma
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) # Updates the memory with the transition event, containing last state, new state, last action played and the last reward
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100: # Comprueba si la memoria tiene suficientes muestras como para aprender y elige 100 muestras
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) # Creates batches of parameters from randomly selected performed actions in memory
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action, self.last_reward, self.done, _ = game.perform_action(action)
        #self.last_action = action
        #self.last_state = new_state
        #self.last_reward = reward
        self.reward_window.append(self.last_reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self): # Computes the mean of the rewards in the reward window
        return sum(self.reward_window)/(len(self.reward_window)+1.)
