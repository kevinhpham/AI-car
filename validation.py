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

def print_action(action):
    """Prints the description of the action based on its index."""
    action_descriptions = {
        0: "Reverse-Left",
        1: "Reverse",
        2: "Reverse-Right",
        3: "Steer-Left (no throttle)",
        4: "No throttle and no steering",
        5: "Steer-Right (no throttle)",
        6: "Forward-right",
        7: "Forward",
        8: "Forward-left"
    }
    print(f"Action {action}: {action_descriptions.get(action, 'Unknown action')}")
def create_bins(num_bins, lower_bounds, upper_bounds):
    """Creates bins for discretizing continuous state space."""
    bins = [np.linspace(lower_bounds[i], upper_bounds[i], num_bins[i] + 1)[1:-1] for i in range(len(num_bins))]
    return bins

def discretize_state(state, bins):
    """Discretizes a continuous state into discrete bins."""
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(bins)))


def load_model_torch(filename="q_learning_model.pt", env=None):
    """Loads the Q-learning model from a file using PyTorch."""
    Q_dict = torch.load(filename)
    print(f"Model loaded from {filename}")
    return defaultdict(default_q_value_factory(env), Q_dict)  # Use the factory function

def main():
# Define the lower and upper bounds of the state space
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True, render_mode='fp_camera')
    env = env.unwrapped
    state, info = env.reset()
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
    else:
        print("No Q file found, creating a new one.")
    # frames = []
    # frames.append(env.render())

    for i in range(200):
        state = discretize_state(state, bins)
        if state in Q:  # Check if the state exists in the Q-table
            action = np.argmax(Q[state])
        else:
            action = 7  # Default action if the state is not found
        print_action(action)
        state, reward, done, _, info = env.step(action)
        # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
        if done:
            break


    env.close()
if __name__ == "__main__":
    main()