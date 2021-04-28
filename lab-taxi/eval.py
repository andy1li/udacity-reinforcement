from agent import Agent
from monitor import interact
import gym
import numpy as np
import pickle

env = gym.make('Taxi-v3')
Q = pickle.load(open('agent.pickle', 'rb'))
print('✅ Agent loaded.')
agent = Agent(Q=Q)
avg_rewards, best_avg_reward = interact(env, agent)
