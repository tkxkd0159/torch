import numpy as np
import matplotlib.pyplot as plt

from kurl.tool import print_values, print_policy, max_dict
from kurl.dp import DP
from kurl.mc import MC
from kurl.td import SARSA, QL


if __name__ == "__main__":

    mydp1 = DP()
    mydp2 = DP()
    mydp1.policy_iteration()
    mydp_changes = mydp2.value_iteration()
    print_values(mydp1.values, mydp1.myenv, 'Policy iterations')
    mydp1.old_policy.update({(0,3): "G", (1,3): "H"})
    mydp1.policy.update({(0,3): "G", (1,3): "H"})
    print_policy(mydp1.old_policy, mydp1.policy, mydp1.myenv, 'Policy iterations')
    print_values(mydp2.values, mydp2.myenv, 'Value iterations')
    mydp2.old_policy.update({(0,3): "G", (1,3): "H"})
    mydp2.policy.update({(0,3): "G", (1,3): "H"})
    print_policy(mydp2.old_policy, mydp2.policy, mydp2.myenv, 'Value iterations')


    mymc = MC()
    mc_deltas = []
    for t in range(100):
        biggest_change = 0
        state_action_returns = mymc.do_episode()
        seen_state_actions =  set()
        for s, a, G in state_action_returns:
            if (s, a) not in seen_state_actions:
                old_q = mymc.Q[s][a]
                mymc.returns[(s, a)].append(G)
                mymc.Q[s][a] = np.mean(mymc.returns[(s, a)])
                biggest_change = max(biggest_change, np.abs(old_q - mymc.Q[s][a]))
                seen_state_actions.add((s, a))
        mc_deltas.append(biggest_change)
        for state in mymc.policy.keys():
            mymc.policy[state] = max_dict(mymc.Q[state])[0]
    mymc.old_policy.update({(0,3): "G", (1,3): "H"})
    mymc.policy.update({(0,3): "G", (1,3): "H"})
    print_policy(mymc.old_policy, mymc.policy, mymc.myenv, "MC")


    mysarsa = SARSA()
    mysarsa.train(100)
    mysarsa.policy.update({(0,3): "G", (1,3): "H"})
    print_policy(None, mysarsa.policy, mysarsa.myenv, "SARSA")


    myql = QL()
    myql.train(100)
    myql.policy.update({(0,3): "G", (1,3): "H"})
    print_policy(None, myql.policy, myql.myenv, "Q-learning")

    t = np.arange(0, 100, 1)
    fig, ax = plt.subplots()
    ax.plot(t, mydp_changes, 'rD', label='Dynamic Programming')
    ax.plot(t, mc_deltas, 'bx-', label='Monte Carlo')
    ax.plot(t, mysarsa.deltas, 'g*', label='SARSA')
    ax.plot(t, mydp_changes, 'y+', label='Q-learning')
    ax.legend()
    ax.set_title("Compare convergence by step")


    plt.show()