import random


def test_td_lambda():
    from src.gridworld import GridWorld
    from src.utils import print_value_function, plot_value_heatmap
    from src.model_Free.td_lambda import td_lambda_prediction

    def random_policy(state):
        return random.choice(env.get_possible_actions(state))

    env = GridWorld()
    V = td_lambda_prediction(env, random_policy, episodes=1000, alpha=0.1, gamma=0.9, lambda_=0.8)

    print("TD(λ) Value Function:")
    print_value_function(V, env.grid_size)
    plot_value_heatmap(V, env.grid_size)
