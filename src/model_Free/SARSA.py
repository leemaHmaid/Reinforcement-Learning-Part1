import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class SARSAAgent:
    def __init__ (self, action_space, gamma =0.9, epsilon = 0.1, alpha = 0.1,decay_rate=0.99):
        self.Q = defaultdict(lambda: np.zeros(action_space))
        self.gamme = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_space = action_space
        self.decay_rate = decay_rate

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return int(np.argmax(self.Q[state]))
    def update_Q(self, state, action , reward, next_state, next_action):
        td_target = reward + self.gamme * self.Q[next_state][next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha*td_error

    def get_greedy_policy(self, terminal_states=None):
        policy = {}
        for state in self.Q:
            if terminal_states and state in terminal_states:
                policy[state] = None
            else:
                policy[state] = int(np.argmax(self.Q[state]))
        return policy
    def train (self, env, num_episodes, decay = False):
        returns_per_episode = []
        episode_lengths = []
        for ep in range(1, num_episodes + 1):
            state = env.reset()
            action = self.epsilon_greedy_action(state)
            done = False
            G = 0
            steps = 0
            while not done:
                next_state, reward, done, _ = env.step_control(action)
                next_action = self.epsilon_greedy_action(next_state)
                G += reward

                self.update_Q(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
                steps +=1
                if decay:
                        self.epsilon = max(0.01, self.epsilon * 0.99)

                if ep % 10000 == 0:
                        print(f"Episode {ep} completed.")
            returns_per_episode.append(G)
            episode_lengths.append(steps)

            if decay:
                self.epsilon = max(0.01, self.epsilon * self.decay_rate)

            if ep % 1000 == 0:
                print(f"Episode {ep} completed.")

        return returns_per_episode, episode_lengths

    
            

            