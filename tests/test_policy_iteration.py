from main import print_policy, print_value_table
from src.dynamic_programming.policy_iteration import policy_iteration
from src.gridworld import GridWorld


def test_policy_iteration():
    env = GridWorld()
    policy, value_table = policy_iteration(env, gamma=0.9, theta=1e-4)

    print_value_table(value_table, env)
    print_policy(policy, env)

 