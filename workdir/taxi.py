import gym

## R,G,Y,B: Possible pickup/destination locations

env = gym.make("Taxi-v2").env

env.reset()

state = env.encode(0,0,0,3) #Taxi row, taxi column, passenger index, destination index
print("State:", state)

env.render()

print(env.P[env.s]) # {action: [(probability, nextstate, reward, done)]}
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
