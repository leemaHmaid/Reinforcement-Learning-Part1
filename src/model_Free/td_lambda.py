import random


def td_lambda_prediction(env, policy, episodes, alpha = 0.1, gamma=0.9, lambda_=0.8):
    """
    TD(λ) prediction algorithm using eligibility traces.
    """

    V ={state: 0.0 for state in env.states}
    for _ in range(episodes):
         E = {state:0.0 for state in env.states}
         state = random.choice(env.states)
         done = env.is_terminal(state)

         while not done:
              action = policy(state)
              transitions = env.get_transition_probabilities(state, action)
              _, next_state, reward = transitions[0]

              td_error = reward + gamma*V[next_state] - V[state]

              E[state] += 1

              for s in env.states:
                   V[s] += alpha * td_error * E[s]
                   E[s] *= gamma * lambda_

              state = next_state
              done = env.is_terminal(state)

    return V
                   