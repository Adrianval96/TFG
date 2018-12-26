import gym
import numpy as np
import tensorflow as tf

env = gym.make("FrozenLake-v0")

inputs = tf.placeholder(shape=[1,16], dtype=tf.float32)

weights = tf.Variable(tf.random_uniform([16,4], 0, 0.1))

Q1 = tf.matmul(inputs, weights)

output = tf.argmax(Q1, 1)

Q2 = tf.placeholder(shape=[1,4], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(Q2-Q1))
gdo = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updatedweights = gdo.minimize(loss)

gamma = 0.9
epsilon = 0.1
episodes = 2000

totalReward = 0

session = tf.Session()
session.run(tf.initialize_all_variables())
for i in range(episodes):
    state_now = env.reset()
    done = False
    reward = 0
    for j in range(100):
        action, Y = session.run([output, Q1], feed_dict = {inputs : [np.eye(16)[state_now]]})
        if epsilon > np.random.rand(1):
            action[0] = env.action_space.sample()
            epsilon -= 10**-3

        #Lets iterate to the next state Note: This can be random.
        state_next , reward, done, _ = env.step(action[0])
        Y1 = session.run(Q1, feed_dict = {inputs : [np.eye(16)[state_next]]})
        change_Y = Y
        change_Y[0, action[0]] = reward + gamma*np.max(Y1)
        #Updating the weights
        _,new_weights = session.run([updatedweights,weights],feed_dict={inputs:[np.eye(16)[state_now]],Q2:change_Y})
        #Lets append the total number of rewards
        totalReward += reward
        state_now = state_next
        if reward == 1:
            print ('Episode {} was successful, Agent reached the Goal'.format(i))

session.close()
