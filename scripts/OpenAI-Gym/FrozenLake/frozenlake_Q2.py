# importing dependency libraries
from __future__ import print_function
import gym as Gym
import numpy as np
import tensorflow as tf
import random

# Load the Environment
env = Gym.make('FrozenLake-v0')


# Q - Network Implementation

## Creating Neural Network

tf.reset_default_graph()
# tensors for inputs, weights, biases, Qtarget
inputs = tf.placeholder(shape=[None,env.observation_space.n],dtype=tf.float32)
W = tf.get_variable(name="W",dtype=tf.float32,shape=[env.observation_space.n,env.action_space.n],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros(shape=[env.action_space.n]),dtype=tf.float32)

qpred = tf.add(tf.matmul(inputs,W),b)
apred = tf.argmax(qpred,1)

qtar = tf.placeholder(shape=[1,env.action_space.n],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(qtar-qpred))

train = tf.train.AdamOptimizer(learning_rate=0.001)
minimizer = train.minimize(loss)

## Training the neural network

init = tf.global_variables_initializer() #initializing tensor variables
#initializing parameters
y = 0.5 #discount factor
e = 0.3 #epsilon value for epsilon-greedy task
episodes = 10000 #total number of episodes

with tf.Session() as sess:
    sess.run(init)
    for i in range(episodes):
        s = env.reset() #resetting the environment at the start of each episode
        r_total = 0 #to calculate the sum of rewards in the current episode
        while(True):
            #running the Q-network created above
            a_pred,q_pred = sess.run([apred,qpred],feed_dict={inputs:np.identity(env.observation_space.n)[s:s+1]})
            #a_pred is the action prediction by the neural network
            #q_pred contains q_values of the actions at current state 's'
            if np.random.uniform(low=0,high=1) < e: #performing epsilon-greedy here
                a_pred[0] = env.action_space.sample()
                #exploring different action by randomly assigning them as the next action
            s_,r,t,_ = env.step(a_pred[0]) #action taken and new state 's_' is encountered with a feedback reward 'r'
            if r==0: 
                if t==True:
                    r=-5 #if hole make the reward more negative
                else:
                    r=-1 #if block is fine/frozen then give slight negative reward to optimize the path
            if r==1:
                r=5 #good positive goat state reward
            
            q_pred_new = sess.run(qpred,feed_dict={inputs:np.identity(env.observation_space.n)[s_:s_+1]})
            #q_pred_new contains q_values of the actions at the new state 
            
            #update the Q-target value for action taken
            targetQ = q_pred
            max_qpredn = np.max(q_pred_new)
            targetQ[0,a_pred[0]] = r + y*max_qpredn
            #this gives our targetQ
            
            #train the neural network to minimize the loss
            _ = sess.run(minimizer,feed_dict={inputs:np.identity(env.observation_space.n)[s:s+1],qtar:targetQ})
            
            s=s_
            if t==True:
                break
    
    #learning ends with the end of the above loop of several episodes above
    #let's check how much our agent has learned
    print("Output after learning")
    print()
    s = env.reset()
    env.render()
    while(True):
        a = sess.run(apred,feed_dict={inputs:np.identity(env.observation_space.n)[s:s+1]})
        s_,r,t,_ = env.step(a[0])
        print("===============")
        env.render()
        s = s_
        if t==True:
            break