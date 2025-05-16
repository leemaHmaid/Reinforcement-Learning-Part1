import random


class GridWorld:
    def __init__(self, grid_size =10 , terminal_states= [(0,0), (9,9)], default_reward = -1, stochastic =False, stochastic_prob = 0.2):
        self.grid_size = grid_size
        self.states = [(i,j) for i in range(grid_size) for j in range(grid_size)]
        self.terminal_states = terminal_states
        self.actions = ['up', 'down', 'left', 'right']
        self.default_reward = default_reward
        self.action_space = len(self.actions)
        self.stochastic = stochastic
        self.stochastic_prob = stochastic_prob

    
    def is_terminal(self, state ):
        return state in self.terminal_states
    
    def get_next_state(self, state, action):

        if self.is_terminal(state):
            return state
        if self.stochastic and random.random() < self.stochastic_prob:
            # Randomly choose a different action
            action = random.choice(self.actions)
        i, j = state
        if action == "up":
            i = max(0, i-1)
        if action == "down":
            i = min(i+1, self.grid_size-1)
        if action == "right":
            j =  min(j+1, self.grid_size-1)
        if action == "left":
            j = max(0, j-1)

        return (i,j)
    
    def get_transition_probabilities(self, state, action):

        """returns list of (prob, next_state, reward) tuples"""
        if self.is_terminal(state):
            return [(1.0, state , 0)]
        next_state = self.get_next_state(state, action)
        return [(1.0, next_state, self.default_reward)]
    
    def get_possible_actions (self, state):
        if self.is_terminal(state):
            return []
        return self.actions
    

    def reset(self):
        #Start in random non-terminal state
        self.current_state = random.choice([s for s in self.states if s  not in self.terminal_states])
        return self.current_state
    
    def step(self, actions):
        if self.is_terminal(self.current_state):
            return self.current_state, 0, True
        action = random.choice(actions)
        next_state = self.get_next_state(self.current_state, action)
        reward = self.default_reward
        done = self.is_terminal(next_state)
        self.current_state = next_state
        return next_state, reward, done
    

    def step_control(self, action):
        if self.is_terminal(self.current_state):
            return self.current_state, 0, True, {}
        
        # Convert action index to string like 'up', 'down', etc.
        action_name = self.actions[action]
        next_state = self.get_next_state(self.current_state, action_name)
        reward = self.default_reward
        done = self.is_terminal(next_state)
        self.current_state = next_state
    
        return next_state, reward, done, {}
