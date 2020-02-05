
import sys
import argparse
from time import sleep
import numpy as np
import tensorflow as tf

##GYM

import gym
from gym import spaces

import vizdoomgym
# importa script drqn a modificar
from drqn import *

#from vizdoomgym.envs.vizdoomdefendcenter import VizdoomDefendCenter as env


## PRUEBA DE CODIGO PARA VIZDOOMGYM ##
#def gymTrain(num_episodes, learning_rate, render = False):
def gymTrain(num_episodes, episode_length, learning_rate, render = False):

    discount_factor = .99

    update_frequency = 5
    store_frequency = 50

    print_frequency = 1000

    total_reward = 0
    total_loss = 0
    old_q_value = 0

    rewards = []
    losses = []

    # TODO INPUT AS PARAMETER
    env = gym.make('VizdoomHealthGathering-v0')
    #env = gym.make(env)


    actions = np.zeros((env.game.get_available_buttons_size(), env.game.get_available_buttons_size()))
    count = 0

    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()

    actionDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size() - 2, learning_rate)
    targetDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size() - 2, learning_rate)

    experiences = ExperienceReplay(1000)

    # for storing the models
    saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    sample = 5
    store = 50

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter("logs", sess.graph)

        sess.run(tf.global_variables_initializer())

        for episode in tqdm.trange(num_episodes, desc="Episode"):
            #done = False
            #frame = 0
            state = env.reset()
            #while not done:
            for frame in tqdm.trange(episode_length, desc="Frame"):
            #for frame in tqdm.trange(env.game.episode_timeout, desc="Frame"):
                #STATE contiene la imagen del juego
                s = state

                #Probablemente hay un problema con la seleccion de la accion porque siempre pilla la misma
                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]
                action = actions[a]
                #print("\nAction = ", action)
                obs, reward, done, info = env.step(action)
                print("Step reward:" + str(reward))
                #total_reward += reward
                tf.summary.scalar('reward', reward)

                if done:
                #if done or frame == episode_length:
                    ep_reward = env.game.get_total_reward()
                    total_reward += ep_reward
                    print(ep_reward)
                    break
                if (frame % store) == 0:
                    experiences.appendToBuffer((s, action, reward))

                if (frame % sample) == 0:
                    memory = experiences.sample(1)
                    mem_frame = memory[0][0]
                    mem_reward = memory[0][2]

                    # network training
                    Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input: mem_frame})
                    Q2 = targetDRQN.output.eval(feed_dict = {targetDRQN.input: mem_frame})

                    # set learning rate
                    learning_rate = actionDRQN.learning_rate.eval()

                    # Calculate Q Value and update
                    Qtarget = old_q_value + learning_rate * (mem_reward + discount_factor * Q2 - old_q_value)
                    old_q_value = Qtarget

                    # Loss function
                    loss = actionDRQN.loss.eval(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    tf.summary.scalar('loss', loss)

                    total_loss += loss

                    # Update networks
                    actionDRQN.update.run(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    targetDRQN.update.run(feed_dict = {targetDRQN.target_vector: Qtarget, targetDRQN.input: mem_frame})

            rewards.append((episode, total_reward))
            tf.summary.scalar('total_reward', total_reward)
            losses.append((episode, total_loss))
            tf.summary.scalar('total_loss', total_loss)

            if episode % 100 == 0:
                saver.save(sess, "./doom_model")

            tqdm.tqdm.write("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))

            total_reward = 0
            total_loss = 0

if __name__ == "__main__":

    #args = parser.parse_args()
    #ray.init()
    #ModelCatalog.register_custom_model("myDoomEnv", DoomEnv) # Registers Doom Model (NOT ENVIRONMENT APPARENTLY)
    #run_single_algo()
    gymTrain(num_episodes=10000, episode_length=100000, learning_rate=0.01, render=False)
    #gymTrain(num_episodes=10000, learning_rate=0.01, render=False)
