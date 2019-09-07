
import sys
import argparse
from time import sleep
import numpy as np
import tensorflow as tf

import time
#from time import time, sleep
from tqdm import trange

import gym
from gym import spaces

from vizdoom import *
import vizdoomgym
# importa script drqn a modificar
from drqn_v2 import *

#from vizdoomgym.envs.vizdoomdefendcenter import VizdoomDefendCenter as env


discount_factor = .99
learning_rate = 0.0005
update_frequency = 200
store_frequency = 15

#How much an episode lasts. has to be changed for each scenario
#TODO
episode_length = 2100
epochs = 10000

print_frequency = 1000


# Training regime
test_episodes_per_epoch = 10

resolution = (160, 256, 3)


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

    pre_train_steps = 5000
    total_steps = 0
    learning_phase = False

    total_reward = 0
    total_loss = 0
    old_q_value = 0

    last_frame = None

    #experiences = ExperienceReplay(1000)

    # for storing the models
    #saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)
    train_episodes_finished = 0

    #for episode in tqdm.trange(epochs, desc="Episode"):
    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_scores = []
        #Las train scores no sirven, hace falta crear estructura de datos
        # compartida por todas las const_eps_epochs para recoger la avg score, max y min


        #print("Training...")
        state = env.reset()
        #print("-----STATE SHAPE DESPUES DE INICIALIZAR")
        #print(state.shape)
        sess.run(tf.global_variables_initializer())

        repeated = frame_repeat

        for learning_step in trange(episode_length, leave=False):

            if total_steps >= pre_train_steps and not learning_phase:
                learning_phase = True
                print("----------BEGINNING TRAINING----------")

            #perform_learning_step(epoch)
            #STATE contiene la imagen del juego
            s_old = state
            start = time.time()
            ##Implementation of frame repeat
            if not learning_phase:
                a = env.action_space.sample()
            elif repeated == frame_repeat:
                ## PROBLEM HERE
                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s_old})[0]
                print("\nSelecting the action took {:.2f} s".format((time.time() - start)))
                repeated = 1
                #curr_action = a
            else:
                #a = curr_action
                repeated += 1
            #print("a: " + str(a))
            action = actions[a]
            #print("\nAction = ", action)
            #start = time.time()
            state, reward, done, info = env.step(action)
            #print("\nA timestep took {:.2f} s".format((time.time() - start)))
            if reward > 0:
                print("Step reward:" + str(reward))
            tf.summary.scalar('reward', reward)

            if done:
            #if done or learning_step == episode_length:
                #print("Episode finished")
                score = env.game.get_total_reward()
                train_scores.append(score)
                state = env.reset()
                train_episodes_finished += 1
                break
            if (learning_step % store_frequency) == 0 or reward > 0:
                experience = np.array([s_old, a, state, done, reward])
                #print("STORING EXPERIENCE")
                experiences.add(np.reshape(experience,[1,5]))
                #experiences.add_transition(s_old, a, state, done, reward)
            if (learning_step % update_frequency) == 0 and learning_phase:
                #THIS WHOLE STEP TAKES A LOT OF TIME AS THE EXECUTION GOES ON
                #print("-----Training from sample-----------")

                start = time.time()
                #memory = experiences.get_sample(1)
                trainBatch = experiences.sample(10)
                #print(trainBatch.shape)

                for i in range (0, len(trainBatch) - 1):
                    mem_frame = trainBatch[i,0]
                    mem_output = trainBatch[i,2]
                    mem_reward = trainBatch[i,4]

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
                t_calc = time.time()-start
                #print("\nCalculate losses and Q updates took {:.2f} s".format((t_calc)))
                # Update networks
                ##HERE'S THE OPERATION THAT TAKES MOST OF THE TIME
                actionDRQN.update.run(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                #t_action = time.time() - start + t_calc
                #print("Update actionDRQN took {:.2f} s".format((t_action)))
                targetDRQN.update.run(feed_dict = {targetDRQN.target_vector: Qtarget, targetDRQN.input: mem_frame})

                print("Update networks took {:.2f} s".format((time.time() - start + t_calc)))
                last_frame = mem_frame

        if learning_phase:
            total_reward = score
            print("TOTAL REWARD: " + str(total_reward))
            rewards.append((epoch, total_reward))
            tf.summary.scalar('total_reward', total_reward)
            losses.append((epoch, total_loss))
            tf.summary.scalar('total_loss', total_loss)

            if epoch % 100 == 0:
                saver.save(sess, "./doom_model")
            print("\n%d training episodes played." % train_episodes_finished)

            #tqdm.tqdm.write("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))
            #print(str(train_scores))
            if (not 'scores' in dir()):
                scores = np.array(train_scores)
            else:
                scores = np.append(scores,train_scores)
            #print(str(train_scores))

            print("Results: mean: %.1f±%.1f," % (scores.mean(), scores.std()), \
                  "min: %.1f," % scores.min(), "max: %.1f," % scores.max())

        total_steps += learning_step
        print("TOTAL STEPS: " + str(total_steps))
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

    env.game.set_window_visible(False)

    #env.game.set_screen_resolution(ScreenResolution.RES_256X160)

    n = env.game.get_available_buttons_size()
    actions = np.zeros((env.game.get_available_buttons_size(), env.game.get_available_buttons_size()))
    count = 0

    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()

    #experiences = ExperienceReplay(1000) ### CODIGO DE JAVI ###
    #experiences = ReplayMemory(capacity=500) ### Ejemplo vizdoom ###
    experiences = experience_buffer(buffer_size=5000)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #actionDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)
    #targetDRQN = DRQN((240, 320, 3), env.game.get_available_buttons_size(), learning_rate)

    actionDRQN = DRQN(resolution, env.game.get_available_buttons_size(), learning_rate)
    targetDRQN = DRQN(resolution, env.game.get_available_buttons_size(), learning_rate)

    saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    #ORIGINAL VALUES
    #sample = 5
    #store = 50

    sample = 25
    store = 10
    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter("logs", sess.graph)

        print("------------Starting the training--------------")

        time_start = time.time()

        gymTrain(epochs, episode_length, learning_rate, render=False)

        print("======================================")
        print("Training finished. It's_old time to watch!")
        # Reinitialize the game with window visible
        #env.game.set_window_visible(True)
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
