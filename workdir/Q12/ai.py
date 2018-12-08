# AI to hack Q12
# This file is meant to implement a convolutional neural network to recognise if there is
# a question on an input image and, if it is, implement text recognition for the question and the answers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Hay que crear ejecutables distintos de experience replay y procesamiento de imagen


class CNN(nn.Module):

    def __init__(self, input_size, nb_outputs):
        super(CNN, self).__init__()

        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)

        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = isquestion)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SoftmaxBody(nn.Module):

    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions

class AI:

    def __init__(self, brain, body):
        self.brain = last_brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        seturn actions.data.numpy()

isquestion = 2 # Possibilities, if it is a question or not

cnn = CNN(isquestion)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# TODO: set up experience replay referencing to created class
################################################################

def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)
