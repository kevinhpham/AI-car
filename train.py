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



def main():
    if __name__ == "__main__":
        # metrics for displaying training status
        os.makedirs("weights", exist_ok=True)
        episode_history = []
        episode_reward_history = []
        env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
        env = env.unwrapped
        # set manual seeds so we get same behaviour everytime - so that when you change your hyper parameters you can attribute the effect to those changes
        episode_batch_score = 0
        episode_reward = 0
        agent = DQN_Solver(env)  # create DQN agent
    # Load checkpoint if it exists
        checkpoint_path = "weights/training_checkpoint.pth"
        start_episode = load_checkpoint(agent, checkpoint_path)

        for i in range(start_episode, EPISODES):
            state, info = env.reset()  # this needs to be called once at the start before sending any actions
            steps = 0
            while True or steps < 500:
                steps += 1
                # sampling loop - sample random actions and add them to the replay buffer
                action = agent.choose_action(state)
                state_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ####### add sampled experience to replay buffer ##########
                agent.memory.add(state,action,reward,state_,done)
                ##########################################################

                # only start learning once replay memory reaches REPLAY_START_SIZE
                if agent.memory.mem_count > REPLAY_START_SIZE:
                    agent.learn()

                state = state_
                episode_batch_score += reward
                episode_reward += reward

                if done:
                    break

            episode_history.append(i)
            episode_reward_history.append(episode_reward)
            episode_reward = 0.0

            # save our model every batches of 100 episodes so we can load later. (note: you can interrupt the training any time and load the latest saved model when testing)
            if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
                torch.save({
                    'policy_network_state_dict': agent.policy_network.state_dict(),
                    'optimizer_state_dict': agent.policy_network.optimizer.state_dict(),
                    'episode': i,
                    'replay_buffer': agent.memory  # Save replay buffer if needed
                }, "weights/training_checkpoint.pth")
    
                print("average total reward per episode batch since episode ", i, ": ", episode_batch_score/ float(100))
                episode_batch_score = 0
            elif agent.memory.mem_count < REPLAY_START_SIZE:
                print("waiting for buffer to fill...")
                episode_batch_score = 0
    env.close()
if __name__ == "__main__":    
    main()