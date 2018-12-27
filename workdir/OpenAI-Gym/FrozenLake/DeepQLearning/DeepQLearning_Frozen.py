#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: adrianval96
"""

import random
import gym
import numpy as np

from datetime import datetime
from argparse import ArgumentParser

import environments.toyText as toyText
import environments.utils as env_utils ### Probablemente no haga falta copiar esto ya que el input es texto

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from utils.params_manager import ParamsManager

from tensorboardX import SummaryWriter

# Esto en principio lo hace para el Q-Learning. En nuestro caso se importa desde parameters.json
MAX_EPISODES = 50000
STEPS_PER_EPISODE = 50
EPSILON_MIN = 0.005
max_num_steps = MAX_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

# Parseador de argumentos
args = ArgumentParser("DeepQLearner_Frozen")
args.add_argument("--params-file", help = "Path del fichero JSON de parámetros. El valor por defecto es parameters.json",
                  default="parameters.json", metavar = "PFILE")
args.add_argument("--env", help = "Entorno de ID de ToyText disponible en OpenAI Gym. El valor por defecto será FrozenLake-v0",
                  default = "FrozenLake-v0", metavar="ENV")
args = args.parse_args()

manager = ParamsManager(args.params_file)

summary_filename_prefix = manager.get_agent_params()['summary_filename_prefix']
summary_filename = summary_filename_prefix + args.env + datetime.now().strftime("%y-%m-%d-%H-%M")

## Summary writer, para visulizar datos de aprendizaje
writer = SummaryWriter(summary_filename)

manager.export_agent_params(summary_filename + "/"+"agent_params.json")
manager.export_environment_params(summary_filename + "/"+"environment_params.json")

#Contador de ejecuciones
global_step_num = 0

use_cuda = manager.get_agent_params()['use_cuda']
device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

# Habilitar la semilla aleatoria para poder reproducir el experimento a posteriori
seed = manager.get_agent_params()['seed']
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

# Queremos introducir nuestra red neuronal en el entrenamiento implementado
# El replay experience ya estará dentro de la clase DeepQLearner
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

class DeepQLearner(object):
    def __init__(self, params):

        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['learning_rate']
        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")
        self.training_steps_completed = 0
        self.action_shape = action_shape

        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate)

        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = self.params['epsilon_max']
        self.epsilon_min = self.params['epsilon_min']
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min,
                                                 max_steps = self.params['epsilon_decay_final_step'])
        self.step_num = 0

        self.memory = ExperienceMemory(capacity = int(self.params['experience_memory_size']))

    ## TODO :: No se muy bien que poner aqui
    #def get_action(self, state):


    def epsilon_greedy_Q(self, state):
        writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)
        self.step_num += 1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(state).data.to(torch.device('cpu')).numpy())
        return action

    def learn(self, state, action, reward, next_state, done):
        if done:
            td_target = reward + 0.0
        else:
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()

    def replay_experience(self, batch_size = None):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size: Tamaño de la muestra a tomar de la memoria
        :return:
        """
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1

    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fragmento de recuerdos anteriores
        :return:
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)/255.0
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)

        if self.params["clip_reward"]:
            reward_batch = np.sign(reward_batch)
        next_obs_batch = np.array(batch_xp.next_obs)/255.0
        done_batch = np.array(batch_xp.done)


        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_frequency'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch *\
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        self.Q_target(next_obs_batch).max(1)[0].data
        else:
            td_target = reward_batch + ~done_batch * \
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        self.Q(next_obs_batch).detach().max(1)[0].data

        td_target = td_target.to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = torch.nn.functional.mse_loss(
                self.Q(obs_batch).gather(1, action_idx.view(-1,1)),
                td_target.float().unsqueeze(1))

        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()

        def save(self, env_name):
            file_name = self.params['save_dir']+"DQL_"+env_name+".ptm"
            agent_state = {"Q": self.Q.state_dict(),
                           "best_mean_reward": self.best_mean_reward,
                           "best_reward": self.best_reward}
            torch.save(agent_state, file_name)
            print("Estado del agente guardado en : ", file_name)


        def load(self, env_name):
            file_name = self.params['load_dir']+"DQL_"+env_name+".ptm"
            agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
            self.Q.load_state_dict(agent_state["Q"])
            self.Q.to(device)
            self.best_mean_reward = agent_state["best_mean_reward"]
            self.best_reward = agent_state["best_reward"]
            print("Cargado del modelo Q desde", file_name,
                  "que hasta el momento tiene una mejor recompensa media de: ",self.best_mean_reward,
                  " y una recompensa máxima de: ", self.best_reward)



## TODO:: crear evento __main__ para lanzar métodos en el orden correcto
## inicializando la NN y creando bucle de aprendizaje

if __name__ == "__main__":
    env_conf = manager.get_environment_params()
    env_conf[env_name] = args.env

    if args.test:
        env_conf["episodic_life"] = False
    reward_type = "LIFE" if env_conf["episodic_life"] else "GAME"

    custom_region_available = False
    for key, value in env_conf["useful_region"].items():
        if key in args.env:
            env_conf["useful_region"] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_conf["useful_region"] = env_conf["useful_region"]["Default"]
    print("Configuración a utilizar:", env_conf)

    toy_env = False
    for game in ToyText.get_games_list():
        if game.replace("_", "") in args.env.lower():
            toy_env = True

    if toy_env:
        environment = ToyText.make_env(args.env, env_conf)
    else:
        environment = env_utils.ResizeReshapeFrames(gym.make(args.env))

    obs_shape = environment.observation_space.shape
    action_shape = environment.action_space.n
    agent_params = manager.get_agent_params()
    agent_params["test"] = args.test
    agent_params["clip_reward"] = env_conf["clip_reward"]
    agent = DeepQLearner(obs_shape, action_shape, agent_params)

    episode_rewards = list()
    previous_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    if agent_params['load_trained_model']:
        try:
            agent.load(env_conf['env_name'])
            previous_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("ERROR: no existe ningún modelo entrenado para este entorno. Empezamos desde cero")


    episode = 0
    while global_step_num < agent_params['max_training_steps']:
        obs = environment.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            if env_conf['render'] or args.render:
                environment.render()

            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            total_reward += reward
            step += 1
            global_step_num += 1

            if done is True:
                episode += 1
                episode_rewards.append(total_reward)

                if total_reward > agent.best_reward:
                    agent.best_reward = total_reward

                if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew:
                    num_improved_episodes_before_checkpoint += 1

                if num_improved_episodes_before_checkpoint >= agent_params['save_freq']:
                    previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0

                print("\n Episodio #{} finalizado con {} iteraciones. Con {} estados: recompensa = {}, recompensa media = {:.2f}, mejor recompensa = {}".
                      format(episode, step+1, reward_type, total_reward, np.mean(episode_rewards), agent.best_reward))

                writer.add_scalar("main/ep_reward", total_reward, global_step_num)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
                writer.add_scalar("main/max_ep_reward", agent.best_reward, global_step_num)

                if agent.memory.get_size() >= 2*agent_params['replay_start_size'] and not args.test:
                    agent.replay_experience()

                break

    environment.close()
    writer.close()
