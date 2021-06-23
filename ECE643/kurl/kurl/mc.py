import numpy as np
from copy import deepcopy

from kurl.env import Grid


class MC:
    def __init__(self):
        self.thresh = 1e-4
        self.gamma = 0.9
        self.possible_actions = ('U', 'D', 'L', 'R')
        self.myenv = Grid.penalty_grid()
        self.states = self.myenv.all_states()

        self.returns = {} # (s, a): G(s, a)
        self.policy = {}
        for state in self.myenv.actions.keys():
            self.policy[state] = np.random.choice(self.possible_actions)
        self.old_policy = deepcopy(self.policy)

        self.Q = {}
        for state in self.states:
            if state in self.myenv.actions:
                self.Q[state] = {}
                for action in self.possible_actions:
                    self.Q[state][action] = 0
                    self.returns[(state, action)] = []
            else:
                pass



    def do_episode(self):
        start_states = list(self.myenv.actions.keys())
        start_idx = np.random.choice(len(start_states))
        self.myenv.set_state(start_states[start_idx])
        s = self.myenv.current_state()
        a = np.random.choice(self.possible_actions)

        states_actions_rewards = [(s, a, 0)]
        seen_states = set()
        seen_states.add(self.myenv.current_state())
        num_steps = 0
        while True:
            r = self.myenv.move(a)
            num_steps += 1
            s = self.myenv.current_state()

            if s in seen_states:
                "와봤던 state에 왔을 경우 penalty"
                r = -10. / num_steps
                states_actions_rewards.append((s, None, r))
                break
            elif self.myenv.game_over():
                states_actions_rewards.append((s, None, r))
                break
            else:
                a = self.policy[s]
                states_actions_rewards.append((s, a, r))
            seen_states.add(s)

        G = 0
        states_actions_returns = []
        first = True
        for s, a, r in reversed(states_actions_rewards):
            if first:
                first = False
            else:
                states_actions_returns.append((s, a, G))
            G = r + self.gamma * G
        states_actions_returns.reverse()
        return states_actions_returns

    @staticmethod
    def max_dict(dict_):
        max_key = None
        max_val = float('-inf')
        for k,v in dict_.items():
            if v > max_val:
                max_val = v
                max_key = k
        return max_key, max_val
