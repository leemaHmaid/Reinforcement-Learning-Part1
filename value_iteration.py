
# Finding the best value Function for a given state in a Markov Decision Process (MDP) using Value Iteration.
# Value Iteration Algorithm
def value_iteration(env, gamma =0.9, theta=1e-3):
    
    V = {state: 0 for state in env.states}
    while True:
        delta = 0
        for state in env.states:
            if env.is_terminal(state):
                continue
            action_values = []
            for action in env.get_possible_actions(state):
                    transtionions = env.get_transition_probabilities(state, action)
                    total = sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward in transtionions)
                    action_values.append(total)

            max_value = max(action_values)
            delta = max(delta, abs(max_value - V[state]))
            V[state] = max_value

        if delta < theta:
              break
        
    return V

# Finf]ing the best policy for a given state in a Markov Decision Process (MDP) using Policy Iteration.

def extract_policy_from_value(env, V, gamma=0.9):
     
     policy = {}
     for state in env.states:
          if env.is_terminal(state):
               continue
          
          best_action =  None
          best_value = float('-inf')

          for action in env.get_possible_actions(state):
               transitions = env.get_transition_probabilities(state, action)
               action_value = sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward in transitions)
               if action_value > best_value:
                    best_value = action_value
                    best_action = action

          policy[state] = best_action

     return policy
                
