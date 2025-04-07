import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import os

# Hyperparameters
EPISODES = 15000                 # Number of episodes to run the training for
LEARNING_RATE = 0.00025         # Learning rate for optimizing the neural network weights
MEM_SIZE = 50000                # Maximum size of the replay memory
REPLAY_START_SIZE = 10000       # Minimum samples in replay memory before learning starts
BATCH_SIZE = 32                 # Number of random samples for training
GAMMA = 0.99                    # Discount factor
EPS_START = 0.1                 # Initial epsilon value for epsilon-greedy action sampling
EPS_END = 0.0001                # Final epsilon value
EPS_DECAY = 4 * MEM_SIZE        # Number of samples over which epsilon decays
MEM_RETAIN = 0.1                # Percentage of initial samples in replay memory to retain
NETWORK_UPDATE_ITERS = 5000     # Steps after which target network weights are updated
FC1_DIMS = 128                  # Number of neurons in the first hidden layer
FC2_DIMS = 128                  # Number of neurons in the second hidden layer

# Neural Network for Q-Learning
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        # Build an MLP with 2 hidden layers
        self.layers = nn.Sequential(
            nn.Linear(*self.input_shape, FC1_DIMS),  # Input layer
            nn.ReLU(),                              # Activation function
            nn.Linear(FC1_DIMS, FC2_DIMS),          # Hidden layer
            nn.ReLU(),                              # Activation function
            nn.Linear(FC2_DIMS, self.action_space)  # Output layer
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # Loss function

    def forward(self, x):
        return self.layers(x)

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)

    def add(self, state, action, reward, state_, done):
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            mem_index = (self.mem_count - int(MEM_SIZE * MEM_RETAIN)) % MEM_SIZE

        self.states[mem_index] = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done

        self.mem_count += 1

    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

# DQN Solver
class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)  # Q
        self.target_network = Network(env)  # \hat{Q}
        self.target_network.load_state_dict(self.policy_network.state_dict())  # Initialize target network
        self.learn_count = 0  # Track the number of learning iterations

    def choose_action(self, observation):
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0

        if random.random() < eps_threshold:
            action = np.random.choice(np.array(range(9)), p=[0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.20, 0.20, 0.20])
            #action= np.random.choice(np.array(range(self.policy_network.action_space)))
            return action

        state = torch.tensor(observation).float().detach().unsqueeze(0)
        self.policy_network.eval()
        with torch.no_grad():
            q_values = self.policy_network(state)
        return torch.argmax(q_values).item()

    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train()
        q_values = self.policy_network(states)[batch_indices, actions]

        self.target_network.eval()
        with torch.no_grad():
            q_values_next = self.target_network(states_)
        q_values_next_max = torch.max(q_values_next, dim=1)[0]

        q_target = rewards + GAMMA * q_values_next_max * dones

        loss = self.policy_network.loss(q_target, q_values)

        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        if self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

def load_checkpoint(agent, checkpoint_path):
    """Loads the training checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        agent.policy_network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1  # Resume from the next episode
        agent.memory = checkpoint['replay_buffer']  # Restore replay buffer if saved
        print(f"Checkpoint loaded. Resuming from episode {start_episode}.")
        return start_episode
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0