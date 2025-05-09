#This will be the main file for the Monte Carlo simulation
# Monte Carlo First visit

from collections import defaultdict

def generate_episodes(env, policy):
    

    """
    Simulate one eposide uisng the given policy
    returns:
         episode: list of (state, action, reward) tuples
    """
    episodes = []
    state = env.reset()
    done = False
    steps = 0

    while not done  and steps < 500:
        action = policy(state)
        next_state, reward, done = env.step(action)
        episodes.append((state, action, reward))
        state = next_state
        steps +=1

    return episodes

def monte_carlo_prediction(env, policy, episodes=500, gamma=0.9, first_visit = False):

    # Returns dictionary of state - estimated value
    V = defaultdict(float)
    returns = defaultdict(list) # returns seen per state

    for  _ in range(episodes):
        if _ % 100 == 0:
            print(f"Episode {_} of {episodes}")
        episode = generate_episodes(env, policy)
        visited_states = set()

        G= 0
        episodes_reserved = list(reversed(episode))

        for _, (state, _ , reward) in enumerate(episodes_reserved):

            G = gamma * G + reward

            if not first_visit and state in visited_states:
                continue
            visited_states.add(state)
            returns[state].append(G)
            V[state]= sum(returns[state]) / len(returns[state])
    return V