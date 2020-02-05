import gym
env=gym.make('DoomCorridor-v0')
state = env.reset()
print(state)
print(env.observation_space.n)

for _ in range(10):
	env.render()
	env.step(env.action_space.sample())
