import random

from src.gridworld import GridWorld
env = GridWorld()
 

def print_value_table(value_table, env):
    print("\n Value Function:")
    for i in range(env.grid_size):
        row = []
        for j in range(env.grid_size):
            state = (i, j)
            v = value_table[state]
            row.append(f"{v:6.2f}")
        print(" ".join(row))

def print_policy(policy, env):
    print("\n Policy:")
    for i in range(env.grid_size):
        row = []
        for j in range(env.grid_size):
            state = (i, j)
            if env.is_terminal(state):
                row.append("  T  ")
            else:
                row.append(f"  {policy[state]}  ")
        print(" ".join(row))



import argparse
import sys

# Add test and source folders to import path
sys.path.append("src")
sys.path.append("tests")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Which test to run (mc, pi, vi, pe, pim)")
    args = parser.parse_args()

    if args.test == "mc":
        from test_monte_carlo import test_monte_carlo
        test_monte_carlo()

    elif args.test == "pi":
        from test_policy_iteration import test_policy_iteration
        test_policy_iteration()

    elif args.test == "vi":
        from test_value_iteration import test_value_iteration
        test_value_iteration()

    elif args.test == "pe":
        from test_policy_evaluation import test_policy_evaluation
        test_policy_evaluation()

    elif args.test == "pim":
        from test_policy_improvement import test_policy_improvement
        test_policy_improvement()
    elif args.test == "td":
        from tests.test_td_prediction import test_td
        test_td()
    elif args.test == "td_lambda":
        from tests.test_td_lambda import test_td_lambda
        test_td_lambda()
    elif args.test == "mc_control":
        from tests.test_mc_control import test_monte_carlo_control
        test_monte_carlo_control(env= env, num_episodes=20000, gamma=0.9, epsilon=0.1, decay=True, verbose=False)


    else:
        print("Unknown test. Use one of: mc, pi, vi, pe, pim, mc_control")

if __name__ == "__main__":
    main()


 