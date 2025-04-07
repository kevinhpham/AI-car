import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
import os

def default_q_value_factory(env):
    def default_q_value():
        return np.zeros(env.action_space.n)
    return default_q_value

def create_bins(num_bins, lower_bounds, upper_bounds):
    """Creates bins for discretizing continuous state space."""
    bins = [np.linspace(lower_bounds[i], upper_bounds[i], num_bins[i] + 1)[1:-1] for i in range(len(num_bins))]
    return bins

def discretize_state(state, bins):
    """Discretizes a continuous state into discrete bins."""
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(bins)))

def save_model_torch(Q, filename="q_learning_model.pt"):
    """Saves the Q-learning model to a file using PyTorch."""
    Q_dict = dict(Q)  # Convert defaultdict to a regular dictionary
    torch.save(Q_dict, filename)
    print(f"Model saved to {filename}")

def load_model_torch(filename="q_learning_model.pt", env=None):
    """Loads the Q-learning model from a file using PyTorch."""
    Q_dict = torch.load(filename)
    print(f"Model loaded from {filename}")
    return defaultdict(default_q_value_factory(env), Q_dict)  # Use the factory function

def epsilon_greedy(env, state, Q, epsilon, episodes, episode):
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = episodes
    sample = np.random.uniform(0, 1)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episode / EPS_DECAY)
    if sample > eps_threshold:
        return np.argmax(Q[state])
    else:
        #action = np.random.choice(np.array(range(9)), p=[0.01, 0.01, 0.01, 0.1, 0.05, 0.1, 0.24, 0.24, 0.24])
        action = env.action_space.sample()
        return action
    
    
def simulate(env, Q, max_episode_length, epsilon, episodes, episode,bins):
    """Rolls out an episode of actions to be used for learning.

    Args:
        env: gym object.
        Q: state-action value function
        epsilon: control how often you explore random actions versus focusing on
                 high value state and actions
        episodes: maximum number of episodes
        episode: number of episodes played so far

    Returns:
        Dataset of episodes for training the RL agent containing states, actions and rewards.
    """
    D = []
    state, _ = env.reset()  # Reset the environment and get the initial state
    state = discretize_state(state, bins)  # Discretize the state
    done = False
    for step in range(max_episode_length):
        action = epsilon_greedy(env, state, Q, epsilon, episodes, episode)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state, bins)  # Discretize the next state
        done = terminated or truncated
        D.append([state, action, reward, next_state])
        state = next_state
        if done:
            break
    return D                                                   # line 10

def q_learning(Q,env, gamma, episodes, max_episode_length, epsilon, step_size,bins):
    """Main loop of Q-learning algorithm.

    Args:
        env: gym object.
        gamma: discount factor - determines how much to value future actions
        episodes: number of episodes to play out
        max_episode_length: maximum number of steps for episode roll out
        epsilon: control how often you explore random actions versus focusing on
                 high value state and actions
        step_size: learning rate - controls how fast or slow to update Q-values
                   for each iteration.

    Returns:
        Q-function which is used to derive policy.
    """
    total_reward = 0
    for episode in range(episodes):                                             # slightly different to line 3, we just run until maximum episodes played out
        D = simulate(env, Q, max_episode_length, epsilon, episodes, episode,bins)    # line 4
        print("episode ", episode)
        for data in D:                                                          # data = [state, action, reward, next_state]  (line 5)
            ####################### update Q value (line 6) #########################
            state, action, reward, next_state = data
            Q[state][action] = (1-step_size)*Q[state][action] + step_size*(reward + gamma*np.max(Q[next_state]))                                                                  # line 6
            #########################################################################
            total_reward += data[2]
        if episode % 50 == 0:
            print("average total reward per episode batch since episode ", episode, ": ", total_reward/ float(50))
            total_reward = 0
            save_model_torch(Q)  # line 8
    return Q  # line 9


def main():

    ######################### renders image from third person perspective for validating policy ##############################
    # env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
    ##########################################################################################################################

    ######################### renders image from onboard camera ###############################################################
    # env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
    ##########################################################################################################################

    ######################### if running locally you can just render the environment in pybullet's GUI #######################
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    env = env.unwrapped
# Define the lower and upper bounds of the state space
    lower_bounds = env.observation_space.low
    upper_bounds = env.observation_space.high

    # Define the number of bins for each dimension
    num_bins = [20] * len(lower_bounds)  # Example: 15 bins for each dimension

    # Create bins
    bins = create_bins(num_bins, lower_bounds, upper_bounds)
    model_file = "q_learning_model.pt"
    if os.path.exists(model_file):
        try:
            Q = load_model_torch(model_file, env)  # Pass env to the function
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new Q-table.")
    else:
        print("No Q file found, creating a new one.")
        Q = defaultdict(default_q_value_factory(env))  # Create a new Q-table
        save_model_torch(Q)
    ##########################################################################################################################
    

    gamma = 0.8             # discount factor - determines how much to value future actions
    episodes = 1000         # number of episodes to play out
    max_episode_length = 200 # maximum number of steps for episode roll out
    epsilon = 0.9               # control how often you explore random actions versus focusing on high value state and actions; high epsilon = explore more
    step_size = 0.15            # learning rate - controls how fast or slow to update Q-values for each iteration.

    state, info = env.reset()
    Q = q_learning(Q, env, gamma, episodes, max_episode_length, epsilon, step_size,bins)  # line 1
    print("Q-learning completed")

    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    env = env.unwrapped
    state, _ = env.reset(0)
    # frames = []
    # frames.append(env.render())

    for i in range(200):
        state = discretize_state(state, bins)
        if state in Q:  # Check if the state exists in the Q-table
            action = np.argmax(Q[state])
        else:
            action = 4  # Default action if the state is not found
        state, reward, done, _, info = env.step(action)
        # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
        if done:
            break

    env.close()
if __name__ == "__main__":
    main()