import matplotlib.pyplot as plt
from IPython import display

def print_values(values, grid, method):
    print(f"\nDisplay values with {method} ")
    for i in range(grid.width):
        print("---------------------")
        for j in range(grid.height):
            v = values.get((i,j),0)
            if v >= 0:
                print(f'{ v:.2f}|', end="")
            else:
                print(f'{v:.2f}|', end="")
        print("")

def print_policy(old_policy, policy, grid, method):
    print(f"\nDisplay actions with {method}")

    if old_policy is not None:
        for i in range(grid.width):
            print("--------------------------------------------------------------")
            for j in range(grid.height):
                old_a = old_policy.get((i, j), ' ')
                print(f'  {old_a}  |', end="")
                if j == len(range(grid.height)) - 1:
                    print(f'  ------->  ', end="")

            for j in range(grid.height):
                a = policy.get((i, j), ' ')
                print(f'  {a}  |', end="")
            print("")

    else:
        for i in range(grid.width):
            print("---------------------------")
            for j in range(grid.height):
                a = policy.get((i, j), ' ')
                print(f'  {a}  |', end="")
            print("")



def max_dict(dict_):
    max_key = None
    max_val = float('-inf')
    for k,v in dict_.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val





def score_plot(scores, mean_scores):
    plt.ion()
    display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)