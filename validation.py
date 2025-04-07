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
from training_config import DQN_Solver, EPISODES, REPLAY_START_SIZE,load_checkpoint

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


def main():
# Define the lower and upper bounds of the state space
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True, render_mode='fp_camera')
    env = env.unwrapped

    agent = DQN_Solver(env)
    checkpoint_path = "weights/training_checkpoint.pth"
    load_checkpoint(agent, checkpoint_path)
    for j in range(4):
        frames = []
        state, info = env.reset()
        agent.policy_network.eval()

        for i in range(200):
            q_values = agent.policy_network(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item() # select action with highest predicted q-value
            print_action(action)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
            if done:
                break

    env.close()
if __name__ == "__main__":
    main()