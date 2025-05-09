import random
from main import print_policy, print_value_table
from src.dynamic_programming.policy_evaluation import policy_evaluation
from src.dynamic_programming.policy_improvement import policy_improvement
from src.gridworld import GridWorld


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