from copy import deepcopy
import numpy as np

from kurl.env import Grid

class DP:
    def __init__(self):
        self.thresh = 1e-4
        self.gamma = 0.9
        self.possible_actions = ('U', 'D', 'L', 'R')
        self.myenv = Grid.penalty_grid()

        self.policy = {}
        for state in self.myenv.actions.keys():
            self.policy[state] = np.random.choice(self.possible_actions)
        self.old_policy = deepcopy(self.policy)
        self.values = {}
        self.states = self.myenv.all_states()
        for state in self.states:
            if state in self.myenv.actions:
                self.values[state] = np.random.random()
            else:
                self.values[state] = 0

    def policy_iteration(self):
        while True:
            #policy evaluation step
            while True:
                biggest_change = 0
                for s in self.states:
                    old_v = self.values[s]

                    if s in self.policy:
                        a = self.policy[s]
                        self.myenv.set_state(s)
                        r = self.myenv.move(a)
                        self.values[s] = r + self.gamma * self.values[self.myenv.current_state()]
                        biggest_change = max(biggest_change, np.abs(old_v - self.values[s]))
                if biggest_change < self.thresh:
                    break

            #policy improvement step
            is_policy_convereged = True
            for s in self.states:
                if s in self.policy:
                    old_a = self.policy[s]
                    new_a = None
                    best_value = float('-inf')
                    for a in self.possible_actions:
                        self.myenv.set_state(s)
                        r = self.myenv.move(a)
                        v = r + self.gamma * self.values[self.myenv.current_state()]
                        if v > best_value:
                            best_value = v
                            new_a = a
                    self.policy[s] = new_a
                    if new_a != old_a:
                        is_policy_convereged = False
            if is_policy_convereged:
                break


    def value_iteration(self):
        while True:
            biggest_change = 0
            for s in self.states:
                old_v = self.values[s]

                if s in self.policy:
                    new_v = float('-inf')

                    for a in self.possible_actions:
                        self.myenv.set_state(s)
                        r = self.myenv.move(a)
                        v = r + self.gamma * self.values[self.myenv.current_state()]
                        if v > new_v:
                            new_v = v
                    self.values[s] = new_v
                    biggest_change = max(biggest_change, np.abs(old_v - self.values[s]))

            if biggest_change < self.thresh:
                break

        for s in self.policy.keys():
            best_a = None
            best_value = float('-inf')
            for a in self.possible_actions:
                self.myenv.set_state(s)
                r = self.myenv.move(a)
                v = r + self.gamma * self.values[self.myenv.current_state()]
                if v > best_value:
                     best_value = v
                     best_a = a
            self.policy[s] = best_a
