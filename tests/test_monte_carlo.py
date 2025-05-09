import random
 
from src.gridworld import GridWorld
from src.model_Free.monte_carlo import monte_carlo_prediction

def random_policy(state):
    env = GridWorld()
    actions = env.get_possible_actions(state)
    return random.choice(actions)

def test_monte_carlo():
    env = GridWorld()

    V = monte_carlo_prediction(env, random_policy, episodes=500, gamma=0.9, first_visit=False)
    for i in range(env.grid_size):
        row = [round(V[(i, j)], 2) for j in range(env.grid_size)]
        print(row)



    