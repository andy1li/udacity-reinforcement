from agent import Agent
from monitor import interact
import gym
import numpy as np
import pickle

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=40000)
pickle.dump(dict(agent.Q), open('agent.pickle', 'wb'))
print('✅ Agent saved.')