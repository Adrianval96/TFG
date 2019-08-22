
import sys
import argparse
from time import sleep
import numpy as np
import tensorflow as tf

from time import time, sleep
from tqdm import trange

import gym
from gym import spaces

import vizdoomgym
# importa script drqn a modificar
from drqn_v2 import *

#from vizdoomgym.envs.vizdoomdefendcenter import VizdoomDefendCenter as env


discount_factor = .99
learning_rate = 0.0005
update_frequency = 5
store_frequency = 50

#How much an episode lasts. has to be changed for each scenario
#TODO
episode_length = 2100
epochs = 10000

print_frequency = 1000


# Training regime
test_episodes_per_epoch = 10


rewards = []
losses = []

frame_repeat = 12
#resolution = (30, 45)
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

    last_frame = None

    #experiences = ExperienceReplay(1000)

    # for storing the models
    #saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    #for episode in tqdm.trange(epochs, desc="Episode"):
    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0

        #Las train scores no sirven, hace falta crear estructura de datos
        # compartida por todas las const_eps_epochs para recoger la avg score, max y min
        train_scores = []

        #print("Training...")
        state = env.reset()
        sess.run(tf.global_variables_initializer())

        for learning_step in trange(episode_length, leave=False):

            #perform_learning_step(epoch)
            #STATE contiene la imagen del juego
            s_old = state
            #print("----------INPUT---------" + str(actionDRQN.input))
            #Probablemente hay un problema con la seleccion de la accion porque siempre pilla la misma
            a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s_old})[0]
            #print("a: " + str(a))
            action = actions[a]
            #print("\nAction = ", action)
            state, reward, done, info = env.step(action)
            if reward != 0:
                print("Step reward:" + str(reward))
            #total_reward += reward
            tf.summary.scalar('reward', reward)

            #if(s_old == state).all():
                #print("-----------LOS ESTADOS SON IGUALES-----------------")
            #print(state)

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
                #experiences.appendToBuffer((s_old, action, reward))
                experiences.add_transition(s_old, a, state, done, reward)
                #experiences.add_transition(s_old, action, state, done, reward)
            if (learning_step % sample) == 0:
                memory = experiences.get_sample(1)

                #print("Memory: " + str(memory[1]))
                #print("Memory: " + str(memory[0]))
                i = memory[5]
                #mem_frame = memory[0][i]
                mem_frame = memory[0].reshape(240, 320, 3)

                #if (mem_frame == last_frame).all():
                #    print("---------------------------------")
                #    print("OJOCUIDAO QUE LOS FRAMES SON IGUALES")
                #    print("---------------------------------")

                mem_output = memory[2]
                mem_reward = memory[4]

                #print("_------------------------------------------------_")
                #print(mem_frame)
                #print("_------------------------------------------------_")

                # network training

                Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input: mem_frame})
                #Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input: mem_frame, actionDRQN.output: mem_output})
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

                last_frame = mem_frame
        total_reward = score
        rewards.append((epoch, total_reward))
        tf.summary.scalar('total_reward', total_reward)
        losses.append((epoch, total_loss))
        tf.summary.scalar('total_loss', total_loss)

        if epoch % 100 == 0:
            saver.save(sess, "./doom_model")
        print("\n%d training episodes played." % train_episodes_finished)

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

            a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s_old})[0]
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

    #experiences = ExperienceReplay(1000) ### CODIGO DE JAVI ###
    experiences = ReplayMemory(capacity=1000) ### Ejemplo vizdoom ###

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    actionDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)
    targetDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)

    #actionDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)
    #targetDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)

    saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    sample = 5
    store = 50

    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter("logs", sess.graph)

        print("------------Starting the training--------------")

        time_start = time()

        gymTrain(epochs, episode_length, learning_rate, render=False)

        print("======================================")
        print("Training finished. It's_old time to watch!")
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
                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s_old})[0]

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
