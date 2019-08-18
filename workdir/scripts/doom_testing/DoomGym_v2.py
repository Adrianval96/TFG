
import sys
import argparse
from time import sleep
import numpy as np
import tensorflow as tf

from time import time, sleep

from tqdm import trange


#from DRQN-tensorflow import src.agent.BaseAgent
#from DRQN-tensorflow import src.replay_memory.DRQNReplayMemory
#from DRQN-tensorflow import src.networks.drqn.DRQN

##GYM

import gym
from gym import spaces

import vizdoomgym
# importa script drqn a modificar
from drqn_v2 import *

#from vizdoomgym.envs.vizdoomdefendcenter import VizdoomDefendCenter as env


discount_factor = .99
learning_rate = 0.00025
update_frequency = 5
store_frequency = 50

#How much an episode lasts. has to be changed for each scenario
#TODO
episode_length = 2100
epochs = 10000

print_frequency = 1000


# Training regime
test_episodes_per_epoch = 100


rewards = []
losses = []

######### LEARNING_TENSORFLOW WITH VIZDOOM CODE ##########
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10
save_model = True
load_model = False
skip_learning = False

# Configuration file path
DEFAULT_MODEL_SAVEFILE = "/tmp/model"
DEFAULT_CONFIG = "../../scenarios/simpler_basic.cfg"



#def gymTrain(epochs, learning_rate, render = False):
def gymTrain(epochs, episode_length, learning_rate, render = False):

    total_reward = 0
    total_loss = 0
    old_q_value = 0

    #experiences = ExperienceReplay(1000)

    # for storing the models
    #saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    #for episode in tqdm.trange(epochs, desc="Episode"):
    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        state = env.reset()
        sess.run(tf.global_variables_initializer())

        for learning_step in trange(episode_length, leave=False):

            #perform_learning_step(epoch)
            #STATE contiene la imagen del juego
            s = state
            #print("----------INPUT---------" + str(actionDRQN.input))
            #Probablemente hay un problema con la seleccion de la accion porque siempre pilla la misma
            a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]
            #print("a: " + str(a))
            action = actions[a]
            print("\nAction = ", action)
            state, reward, done, info = env.step(action)
            print("Step reward:" + str(reward))
            #total_reward += reward
            tf.summary.scalar('reward', reward)

            if done:
            #if done or learning_step == episode_length:
                score = env.game.get_total_reward()
                train_scores.append(score)
                state = env.reset()
                train_episodes_finished += 1
                #total_reward += score
                #print(score)
                break
            if (learning_step % store) == 0:
                experiences.appendToBuffer((s, action, reward))

            if (learning_step % sample) == 0:
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

        total_reward = score
        rewards.append((epoch, total_reward))
        tf.summary.scalar('total_reward', total_reward)
        losses.append((epoch, total_loss))
        tf.summary.scalar('total_loss', total_loss)

        if epoch % 100 == 0:
            saver.save(sess, "./doom_model")
        print("%d training episodes played." % train_episodes_finished)

        #tqdm.tqdm.write("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

    print("\nTesting...")
    test_episode = []
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        state = env.reset()
        done = False
        while not done:

            a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]
            #print("---------------------------------------------------------------")
            #print("Action directly from prediction: " + str(a))
            action = actions[a]
            #print("Action from array: " + str(action))

            state, reward, done, info = env.step(action)

            #state, reward, done, info = env.step(action)
            #best_action_index = get_best_action(state)

            #game.make_action(actions[best_action_index], frame_repeat)
        r = env.game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f±%.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())

    print("Saving the network weigths to:", DEFAULT_MODEL_SAVEFILE)
    saver.save(sess, DEFAULT_MODEL_SAVEFILE)

    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))


    #tqdm.tqdm.write("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))

    total_reward = 0
    total_loss = 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mcneto's attempt to get a DRQN network working with vizdoom.")
    #parser.add_argument(dest="config",
    #                    default=DEFAULT_CONFIG,
    #                    nargs="?",
    #                    help="Path to the configuration file of the scenario."
    #                         " Please see "
    #                         "../../scenarios/*cfg for more scenarios.")

    parser.add_argument("--scenario", type=str, help = "Scenario to play in Vizdoom. Default:VizdoomBasic-v0", default = "VizdoomBasic-v0")
    args = parser.parse_args()


    # TODO INPUT AS PARAMETER
    env = gym.make(args.scenario)
    #env = gym.make('VizdoomHealthGathering-v0')

    n = env.game.get_available_buttons_size()
    actions = np.zeros((env.game.get_available_buttons_size(), env.game.get_available_buttons_size()))
    count = 0

    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()

    experiences = ExperienceReplay(1000) ### CODIGO DE JAVI ###
    #experiences = ReplayMemory(capacity=1000) ### Ejemplo vizdoom ###

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    actionDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)
    targetDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)
    #actionDRQN = DRQN((30, 45, 3), env.game.get_available_buttons_size() - 2, learning_rate)
    #targetDRQN = DRQN((30, 45, 3), env.game.get_available_buttons_size() - 2, learning_rate)

    ### CODE DRQN ATARI ###
    #net = DRQN(self.env_wrapper.action_space.n, config)
    #net.build()
    #net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

    saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    sample = 5
    store = 50

    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter("logs", sess.graph)

        print("------------Starting the training--------------")

        time_start = time()
        #if not skip_learning:



        ##TODO
        #args = parser.parse_args()
        #ray.init()
        #ModelCatalog.register_custom_model("myDoomEnv", DoomEnv) # Registers Doom Model (NOT ENVIRONMENT APPARENTLY)
        #run_single_algo()
        gymTrain(epochs, episode_length, learning_rate, render=False)

        print("======================================")
        print("Training finished. It's time to watch!")
        # Reinitialize the game with window visible
        env.game.set_window_visible(True)
        env.game.set_mode(vzd.Mode.ASYNC_PLAYER)
        #game.init()

        for _ in range(episodes_to_watch):
            env.reset()
            done = False
            while not done:
                state, reward, done, info = env.step(action)
                print("---------------------------------------------------------------")
                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]

                action = actions[a]
                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                env.game.set_action(a)
                for _ in range(frame_repeat):
                    env.game.advance_action()

            # Sleep between episodes
            sleep(1.0)
            score = game.get_total_reward()
            print("Total score: ", score)


    #gymTrain(epochs=10000, learning_rate=0.01, render=False)
