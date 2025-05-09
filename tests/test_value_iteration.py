from main import print_policy, print_value_table
from src.dynamic_programming.value_iteration import extract_policy_from_value, value_iteration
from src.gridworld import GridWorld


def test_value_iteration():
    env = GridWorld()
    value_table = value_iteration(env, gamma=0.9, theta=1e-4)

    print_value_table(value_table, env)
    policy = extract_policy_from_value(env, value_table, gamma=0.9)
    print_policy(policy, env)