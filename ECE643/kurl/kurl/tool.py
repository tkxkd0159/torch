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
