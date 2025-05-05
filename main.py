from gridworld import GridWorld
from policy_evaluation import policy_evaluation_No_Discount as evaluate_policy
import random


def main():
    env = GridWorld()

    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            continue
        policy[state] = random.choice(env.get_possible_actions(state))

    value_table = evaluate_policy(env, policy, gamma=0.9, theta=1e-4)

    print("\n🧠 Value Function:")
    for i in range(env.grid_size):
        row = []
        for j in range(env.grid_size):
            state = (i, j)
            v = value_table[state]
            row.append(f"{v:6.2f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()