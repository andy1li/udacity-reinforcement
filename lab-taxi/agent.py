import numpy as np
from collections import defaultdict
import dill

class Agent:

    def __init__(self, nA=6, Q=None, ϵ=0.001, α=0.025, γ=1):
        """ Initialize agent.
    
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.is_training = not Q
        print(f'⌛️ {["Evaluating", "Training"][self.is_training]} agent:')
        self.Q = defaultdict(lambda: np.zeros(self.nA)) if not Q else Q
        self.ϵ, self.α, self.γ = ϵ, α, γ

    def select_action(self, state):
        """ Given the state, select an action.
        (Epsilon-Greedy)

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        q = self.Q[state]
        if self.is_training and (np.random.random() < self.ϵ or all(q == q[0])):
            return np.random.choice(range(self.nA))
        else:
            return np.argmax(q)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        (Max-Q Learning)

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.is_training:
            target = reward + self.γ * max(self.Q[next_state])
            self.Q[state][action] += self.α * (target - self.Q[state][action])