from copy import deepcopy
import numpy as np

from kurl.env import Grid, random_action
from kurl.tool import max_dict

class SARSA:
    def __init__(self):
        self.thresh = 1e-4
        self.gamma = 0.9
        self.alpha = 0.1 # learning rate
        self.possible_actions = ('U', 'D', 'L', 'R')
        self.myenv = Grid.penalty_grid()
        self.policy = {}
        self.values = {}

        self.Q = {}
        self.update_counts = {}
        self.adjust_factor = {}
        self.states = self.myenv.all_states()
        for state in self.states:
            self.Q[state] = {}
            self.adjust_factor[state] = {}
            for action in self.possible_actions:
                self.Q[state][action] = 0
                self.adjust_factor[state][action] = 1.0

        self.deltas = []
        self.biggest_change = 0


    def do(self, epoch: int):
        exploit_factor = 1.0
        for i in range(epoch):
            if i % 100 == 0:
                exploit_factor += 10e-3

            s = self.myenv.start
            self.myenv.set_state(s)

            a = max_dict(self.Q[s])[0]
            a = random_action(a, self.possible_actions, eps=0.5/exploit_factor)

            while not self.myenv.game_over():
                r = self.myenv.move(a)
                s2 = self.myenv.current_state()
                a2 = max_dict(self.Q[s2])[0]
                a2 = random_action(a2, self.possible_actions, eps=0.5/exploit_factor)
                alpha = self.alpha / self.adjust_factor[s][a]
                self.adjust_factor[s][a] += 0.005
                old_q = self.Q[s][a]
                self.Q[s][a] = self.Q[s][a] + alpha * (r + self.gamma * self.Q[s2][a2] - self.Q[s][a])

                self.biggest_change = max(self.biggest_change, np.abs(old_q - self.Q[s][a]))
                self.update_counts[s] = self.update_counts.get(s, 0) + 1

                s = s2
                a = a2

            self.deltas.append(self.biggest_change)

        for s in self.myenv.actions.keys():
            a, max_q = max_dict(self.Q[s])
            self.policy[s] = a
            self.values[s] = max_q






class QL:
    def __init__(self):
        self.thresh = 1e-4
        self.gamma = 0.9
        self.alpha = 0.1 # learning rate
        self.possible_actions = ('U', 'D', 'L', 'R')
        self.myenv = Grid.penalty_grid()
        self.policy = {}
        self.values = {}

        self.Q = {}
        self.update_counts = {}
        self.adjust_factor = {}
        self.states = self.myenv.all_states()
        for state in self.states:
            self.Q[state] = {}
            self.adjust_factor[state] = {}
            for action in self.possible_actions:
                self.Q[state][action] = 0
                self.adjust_factor[state][action] = 1.0

        self.deltas = []
        self.biggest_change = 0


    def do(self, epoch):
        exploit_factor = 1.0
        for i in range(epoch):
            if i % 100 == 0:
                exploit_factor += 10e-3

            s = self.myenv.start
            self.myenv.set_state(s)

            a = max_dict(self.Q[s])[0]

            while not self.myenv.game_over():
                # SARSA랑 달리 업데이트된거 그대로 안쓰고 e-greedy
                a = random_action(a, self.possible_actions, eps=0.5/exploit_factor)
                r = self.myenv.move(a)
                s2 = self.myenv.current_state()
                a2, max_q_s2a2 = max_dict(self.Q[s2])

                alpha = self.alpha / self.adjust_factor[s][a]
                self.adjust_factor[s][a] += 0.005

                old_q = self.Q[s][a]
                self.Q[s][a] = self.Q[s][a] + alpha * (r + self.gamma * max_q_s2a2 - self.Q[s][a])

                self.biggest_change = max(self.biggest_change, np.abs(old_q - self.Q[s][a]))
                self.update_counts[s] = self.update_counts.get(s, 0) + 1

                s = s2
                a = a2

            self.deltas.append(self.biggest_change)

        for s in self.myenv.actions.keys():
            a, max_q = max_dict(self.Q[s])
            self.policy[s] = a
            self.values[s] = max_q