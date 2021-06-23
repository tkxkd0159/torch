import numpy as np
import matplotlib.pyplot as plt

from kurl.tool import print_values, print_policy
from kurl.dp import DP
from kurl.mc import MC





if __name__ == "__main__":

    target = "MC"
    # while target is None:
    #     try:
    #         method = input("Select your algorithms : DP, MC, SARSA, Q, DQ, MCPG, A2C, DDPG \n")
    #         print(f'You select {method}')

    #         if method not in ("DP", "MC", "SARSA", "Q", "DQ", "MCPG", "A2C", "DDPG"):
    #             print("test")
    #             raise ValueError
    #         else:
    #             target = method
    #             break

    #     except ValueError:
    #         print("Please enter a valid option")



    if target == "DP":
        mydp1 = DP()
        mydp2 = DP()
        mydp1.policy_iteration()
        mydp2.value_iteration()

        print_values(mydp1.values, mydp1.myenv, 'Policy iterations')
        mydp1.old_policy.update({(0,3): "G", (1,3): "H"})
        mydp1.policy.update({(0,3): "G", (1,3): "H"})
        print_policy(mydp1.old_policy, mydp1.policy, mydp1.myenv, 'Policy iterations')

        print_values(mydp2.values, mydp2.myenv, 'Value iterations')
        mydp2.old_policy.update({(0,3): "G", (1,3): "H"})
        mydp2.policy.update({(0,3): "G", (1,3): "H"})
        print_policy(mydp2.old_policy, mydp2.policy, mydp2.myenv, 'Value iterations')

    elif target == "MC":
        mymc = MC()
        deltas = []

        for t in range(10000):
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

            deltas.append(biggest_change)
            for state in mymc.policy.keys():
                mymc.policy[state] = MC.max_dict(mymc.Q[state])[0]

        mymc.old_policy.update({(0,3): "G", (1,3): "H"})
        mymc.policy.update({(0,3): "G", (1,3): "H"})
        print_policy(mymc.old_policy, mymc.policy, mymc.myenv, "MC")

        plt.plot(deltas)
        plt.show()

    elif target == "SARSA":
        pass