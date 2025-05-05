from gridworld import GridWorld
from policy_evaluation import policy_evaluation
from policy_improvement import policy_improvement
from policy_iteration import policy_iteration
from value_iteration import value_iteration, extract_policy_from_value

import random

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


def test_gridworld():
    env = GridWorld()
    print("\n States in GridWorld:")
    print(env.states)
    print("\n Possible actions from (1, 1):")
    print(env.get_possible_actions((1, 1)))


def test_policy_evaluation():
    env = GridWorld()

    #  Build a random policy for all non-terminal states
    policy = {}
    for state in env.states:
        if env.is_terminal(state):
            continue
        actions = env.get_possible_actions(state)
        if actions:
            policy[state] = random.choice(actions)

    # 🔍 Evaluate the policy
    value_table = policy_evaluation(env, policy, gamma=0.9, theta=1e-4)

    # Display result
    print_value_table(value_table, env)
    print_policy(policy, env)


def test_policy_improvement():
    env = GridWorld()

    # Start from a random policy
    policy = {}
    for state in env.states:
        if env.is_terminal(state):
            continue
        actions = env.get_possible_actions(state)
        if actions:
            policy[state] = random.choice(actions)

    # Evaluate and improve
    value_table = policy_evaluation(env, policy, gamma=0.9, theta=1e-4)
    improved_policy = policy_improvement(env, value_table, gamma=0.9)

    # Show results
    print_value_table(value_table, env)
    print_policy(improved_policy, env)

def test_policy_iteration():
    env = GridWorld()
    policy, value_table = policy_iteration(env, gamma=0.9, theta=1e-4)

    print_value_table(value_table, env)
    print_policy(policy, env)

def test_value_iteration():
    env = GridWorld()
    value_table = value_iteration(env, gamma=0.9, theta=1e-4)

    print_value_table(value_table, env)
    policy = extract_policy_from_value(env, value_table, gamma=0.9)
    print_policy(policy, env)



if __name__ == "__main__":
    # Uncomment what you want to test
    # test_gridworld()
    # test_policy_evaluation()
    # test_policy_improvement()
    #  test_policy_iteration()
     test_value_iteration()
