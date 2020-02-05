import tensorflow as tf
import numpy as np
import math
#from vizdoom import *
import timeit
import math
import os
import sys
import tqdm
import bcolz

from random import sample, randint, random

import math

resolution = (160, 256)
#resolution = (240, 320)

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def get_input_shape(Image,Filter,Stride):
    layer1 = math.ceil(((Image - Filter + 1) / Stride))

    o1 = math.ceil((layer1 / Stride))

    layer2 = math.ceil(((o1 - Filter + 1) / Stride))

    o2 = math.ceil((layer2 / Stride))

    layer3 = math.ceil(((o2 - Filter + 1) / Stride))

    o3 = math.ceil((layer3  / Stride))

    #print("---------------INPUT SHAPE-----------------")
    #print(str(o3))

    return int(o3)

class DRQN():
    def __init__(self, input_shape, num_actions, inital_learning_rate):

        # first, we initialize all the hyperparameters

        self.tfcast_type = tf.float32
        # shape of our input which would be (length, width, channels)
        self.input_shape = input_shape
        # number of actions in the environment
        self.num_actions = num_actions
        # learning rate for the neural network
        self.learning_rate = inital_learning_rate

        # now we will define the hyperparameters of the convolutional neural network

        # filter size
        #self.filter_size = 20
        self.filter_size = 5

        # number of filters
        #self.num_filters = [64, 32, 16]
        self.num_filters = [16, 32, 64]

        # stride size
        self.stride = 2

        # pool size
        self.poolsize = 2

        # shape of our convolutional layer
        self.convolution_shape = get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]

        # now we define the hyperparameters nof our recurrent neural network and the final feed forward layer

        # number of neurons
        self.cell_size = 50

        # drop out probability
        self.dropout_probability = [0.3, 0.2]

        # hyperparameters for optimization
        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180


        # initialize all the variables for the CNN

        # we initialize the placeholder for input whose shape would be (length, width, channel)
        self.input = tf.placeholder(shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = self.tfcast_type)
        #print("-----------------------------------------------")
        #print(tf.shape(input))
        #print("-----------------------------------------------")
        # we will also initialize the shape of the target vector whose shape is equal to the number of actions
        self.target_vector = tf.placeholder(shape = (self.num_actions, 1), dtype = self.tfcast_type)

        # initialize feature maps for our corresponding 3 filters
        self.features1 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, input_shape[2], self.num_filters[0]),
                                     dtype = self.tfcast_type)

        self.features2 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]),
                                     dtype = self.tfcast_type)

        self.features3 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]),
                                     dtype = self.tfcast_type)

        # initialize variables for RNN

        self.h = tf.Variable(initial_value = np.zeros((1, self.cell_size)), dtype = self.tfcast_type)

        # input to hidden weight matrix
        self.rU = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type)


        # hidden to hidden weight matrix
        self.rW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            high = np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            size = (self.convolution_shape, self.cell_size)),
                              dtype = self.tfcast_type)


        # hidden to output weight matrix

        self.rV = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type)
        # bias
        self.rb = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)
        self.rc = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)


        # initialize weights and bias of feed forward network

        # weights
        self.fW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            high = np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            size = (self.cell_size, self.num_actions)),
                              dtype = self.tfcast_type)

        # bias
        self.fb = tf.Variable(initial_value = np.zeros(self.num_actions), dtype = self.tfcast_type)

        # learning rate
        self.step_count = tf.Variable(initial_value = 0, dtype = self.tfcast_type)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.step_count,
                                                   self.loss_decay_steps,
                                                   self.loss_decay_steps,
                                                   staircase = False)


        # now let us build the network

        # first convolutional layer
        self.conv1 = tf.nn.conv2d(input = tf.reshape(self.input, shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), filter = self.features1, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = tf.nn.max_pool(self.relu1, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # second convolutional layer
        self.conv2 = tf.nn.conv2d(input = self.pool1, filter = self.features2, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = tf.nn.max_pool(self.relu2, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # third convolutional layer
        self.conv3 = tf.nn.conv2d(input = self.pool2, filter = self.features3, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu3 = tf.nn.relu(self.conv3)
        self.pool3 = tf.nn.max_pool(self.relu3, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # add dropout and reshape the input
        self.drop1 = tf.nn.dropout(self.pool3, self.dropout_probability[0])
        self.reshaped_input = tf.reshape(self.drop1, shape = [1, -1])
        #print(self.reshaped_input)

        # now we build recurrent neural network which takes the input from the last layer of convolutional network
        self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU) + self.rb)
        self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc)

        # add drop out to RNN
        self.drop2 = tf.nn.dropout(self.o, self.dropout_probability[1])

        # we feed the result of RNN to the feed forward layer
        self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb, shape = [-1, 1])
        self.prediction = tf.argmax(self.output)

        # compute loss
        self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output))

        # we use Adam optimizer for minimizing the error
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # compute gradients of the loss and update the gradients
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.update = self.optimizer.apply_gradients(self.gradients)

        self.parameters = (self.features1, self.features2, self.features3,
                           self.rW, self.rU, self.rV, self.rb, self.rc,
                           self.fW, self.fb)


class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        selected = np.array(sample(self.buffer, size))
        #print("SAMPLE: " + str(selected))
        #print("SAMPLE SHAPE: " + str(selected.shape))
        #print("--------------------------------------")
        #print(str(selected[0],))
        return np.reshape(selected, [size, 5])




class ReplayMemory():
    def __init__(self, capacity):
        channels = 3
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

        self.memoryUsed = None

    def add_transition(self, s1, action, s2, isterminal, reward):
        #print("-------------------------SHAPES IN ADD TRANSITION--------------------------")
        #print(s1.shape)
        #print(self.s1.shape)
        #print("-------------------------SHAPES IN ADD TRANSITION--------------------------")
        self.s1[self.pos, :, :, :] = s1
        #self.s1[self.pos, :, :, 0] = s1
        #print(str(action))
        self.a[self.pos] = action
        #self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
            #self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward


        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        #memoryUsed = convert_size(sys.getsizeof(self.))
        #if self.memoryUsed != memoryUsed:
        #    self.memoryUsed = memoryUsed
        #    tqdm.tqdm.write("Experience Replay size: {}".format(memoryUsed))


    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        #print("----------------------------_" + str(self.s1[i]))
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i], i
