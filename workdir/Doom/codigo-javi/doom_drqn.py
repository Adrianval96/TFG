import tensorflow as tf
import numpy as np
import math
from vizdoom import *
import timeit
import math
import os
import sys
import tqdm
import bcolz

import math

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
        self.filter_size = 5
        
        # number of filters
        self.num_filters = [16, 32, 64]
        
        # stride size
        self.stride = 2
        
        # pool size
        self.poolsize = 2        
        
        # shape of our convolutional layer
        self.convolution_shape = get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]
        
        # now we define the hyperparameters of our recurrent neural network and the final feed forward layer
        
        # number of neurons 
        self.cell_size = 100
        
        # number of hidden layers
        self.hidden_layer = 50
        
        # drop out probability
        self.dropout_probability = [0.3, 0.2]

        # hyperparameters for optimization
        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180

        
        # initialize all the variables for the CNN

        # we initialize the placeholder for input whose shape would be (length, width, channel)
        self.input = tf.placeholder(shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = self.tfcast_type)
        
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
        # recall how RNN works from chapter 7
        
        self.h = tf.Variable(initial_value = np.zeros((1, self.cell_size)), dtype = self.tfcast_type)
        
        # hidden to hidden weight matrix
        self.rW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            high = np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            size = (self.convolution_shape, self.cell_size)),
                              dtype = self.tfcast_type)
        
        # input to hidden weight matrix
        self.rU = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type)
        
        # hiddent to output weight matrix
                          
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
        
class ExperienceReplay():
    def __init__(self, buffer_size):
        
        # buffer for holding the transition 
        self.buffer = []
        self.size = None
        
        # size of the buffer
        self.buffer_size = buffer_size
        
    # we remove the old transition if buffer size has reached its limit. Think of the buffer as a queue when new
    # one comes, old one goes off
    
    def appendToBuffer(self, memory_tuplet):
        memory_tuplet = (bcolz.carray(memory_tuplet[0]), memory_tuplet[1], memory_tuplet[2])
        if len(self.buffer) > self.buffer_size: 
            for i in range(len(self.buffer) - self.buffer_size):
                self.buffer.remove(self.buffer[0])     
        self.buffer.append(memory_tuplet)
        size = convert_size(sys.getsizeof(self.buffer))
        if self.size != size:
            self.size = size
            tqdm.tqdm.write("Experience Replay size: {}".format(size))
        
        
    # define a function called sample for sampling some random n number of transistions   
    
    def sample(self, n):
        memories = []
        
        for i in range(n):
            memory_index = np.random.randint(0, len(self.buffer))       
            memories.append(self.buffer[memory_index])
        return memories
        
def train(num_episodes, episode_length, learning_rate, scenario = "deathmatch.cfg", map_path = 'map02', render = False):
  
    # discount parameter for Q-value computation
    discount_factor = .99
    
    # frequency for updating the experience in the buffer
    update_frequency = 5
    store_frequency = 50
    
    # for printing the output
    print_frequency = 1000

    # initialize variables for storing total rewards and total loss
    total_reward = 0
    total_loss = 0
    old_q_value = 0

    # initialize lists for storing the episodic rewards and losses 
    rewards = []
    losses = []

    # okay, now let us get to the action!
   
    # first, we initialize our doomgame environment
    game = DoomGame()
    
    # specify the path where our scenario file is located
    game.set_doom_scenario_path(scenario)
    
    # specify the path of map file
    game.set_doom_map(map_path)

    # then we set screen resolution and screen format
    game.set_screen_resolution(ScreenResolution.RES_256X160)    
    game.set_screen_format(ScreenFormat.RGB24)

    # we can add particles and effetcs we needed by simply setting them to true or false
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)

    # now we will specify buttons that should be available to the agent
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.MOVE_BACKWARD)
    game.add_available_button(Button.ATTACK)
    
   
    # okay,now we will add one more button called delta. The above button will only work 
    # like a keyboard keys and will have only boolean values. 

    # so we use delta button which emulates a mouse device which will have positive and negative values
    # and it will be useful in environment for exploring
    
    game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA, 90)
    game.add_available_button(Button.LOOK_UP_DOWN_DELTA, 90)

    # initialize an array for actions
    actions = np.zeros((game.get_available_buttons_size(), game.get_available_buttons_size()))
    count = 0
    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()


    # then we add the game variables, ammo, health, and killcount
    game.add_available_game_variable(GameVariable.AMMO0)
    game.add_available_game_variable(GameVariable.HEALTH)
    game.add_available_game_variable(GameVariable.KILLCOUNT)

    # we set episode_timeout to terminate the episode after some time step
    # we also set episode_start_time which is useful for skipping intial events
    
    game.set_episode_timeout(6 * episode_length)
    game.set_episode_start_time(10)
    game.set_window_visible(render)
    
    # we can also enable sound by setting set_sound_enable to true
    game.set_sound_enabled(False)

    # we set living reward to 0 which the agent for each move it does even though the move is not useful
    game.set_living_reward(0)

    # doom has different modes such as player, spectator, asynchronous player and asynchronous spectator
    
    # in spectator mode humans will play and agent will learn from it.
    # in player mode, agent actually plays the game, so we use player mode.
    
    game.set_mode(Mode.PLAYER)

    # okay, So now we, initialize the game environment
    game.init()

    # now, let us create instance to our DRQN class and create our both actor and target DRQN networks
    actionDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)
    targetDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)
    
    # we will also create instance to the ExperienceReplay class with the buffer size of 1000
    experiences = ExperienceReplay(1000)

    # for storing the models
    saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    
    # now let us start the training process
    # we initialize variables for sampling and storing transistions from the experience buffer
    sample = 5
    store = 50
   
    # start the tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #sess = tf.Session(config=config)
    #with tf.Session() as sess:
    with tf.Session(config=config) as sess:
        
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter("logs", sess.graph)
        
        # initialize all tensorflow variables
        
        sess.run(tf.global_variables_initializer())
        
        for episode in tqdm.trange(num_episodes, desc="Episode"):
            
            # start the new episode
            game.new_episode()
            
            # play the episode till it reaches the episode length
            for frame in tqdm.trange(episode_length, desc="Frame"):
                
                # get the game state
                state = game.get_state()
                s = state.screen_buffer
                
                # select the action
                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]
                action = actions[a]
                
                # perform the action and store the reward
                reward = game.make_action(action)
                print("Action = ", action)
                
                # update total rewad
                total_reward += reward
                tf.summary.scalar('reward', reward)

               
                # if the episode is over then break
                if game.is_episode_finished():
                    break
                 
                # store transistion to our experience buffer
                if (frame % store) == 0:
                    experiences.appendToBuffer((s, action, reward))

                # sample experience form the experience buffer        
                if (frame % sample) == 0:
                    memory = experiences.sample(1)
                    mem_frame = memory[0][0]
                    mem_reward = memory[0][2]
                    
                    
                    # now, train the network
                    Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input: mem_frame})
                    Q2 = targetDRQN.output.eval(feed_dict = {targetDRQN.input: mem_frame})

                    # set learning rate
                    learning_rate = actionDRQN.learning_rate.eval()

                    # calculate Q value
                    Qtarget = old_q_value + learning_rate * (mem_reward + discount_factor * Q2 - old_q_value)    
                    
                    # update old Q value
                    old_q_value = Qtarget

                    # compute Loss
                    loss = actionDRQN.loss.eval(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    tf.summary.scalar('loss', loss)
                    
                    # update total loss
                    total_loss += loss

                    # update both networks
                    actionDRQN.update.run(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    targetDRQN.update.run(feed_dict = {targetDRQN.target_vector: Qtarget, targetDRQN.input: mem_frame})

            rewards.append((episode, total_reward))
            tf.summary.scalar('total_reward', total_reward)
            losses.append((episode, total_loss))
            tf.summary.scalar('total_loss', total_loss)
            
            if episode % 100 == 0:
                saver.save(sess, "./doom_model")

            
            #print("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))
            tqdm.tqdm.write("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))


            total_reward = 0
            total_loss = 0


train(num_episodes=10000, episode_length=300, learning_rate=0.01, render=False)
