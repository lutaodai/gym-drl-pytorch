import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.replay_buffer import ReplayBuffer


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, args, device):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = args.seed
        self.double_dqn = args.double_dqn
        self.dueling_dqn = args.dueling_dqn
        self.args = args
        self.device = device
        assert self.double_dqn * self.dueling_dqn == 0
        if self.double_dqn:
            print("Implementing Double DQN!")
        elif self.dueling_dqn:
            print("Implementing Dueling DQN!")
        else:
            print("Implementing DQN")

        # Q-Network
        if self.dueling_dqn:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, self.seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, self.seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, self.seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, self.seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.args.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, args.buffer_size, args.batch_size, self.seed, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_FREQUENCY time steps.
        self.t_step = (self.t_step + 1) % self.args.update_frequency
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.args.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.args.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        if self.double_dqn:
            next_actions = torch.argmax(self.qnetwork_local(next_states), dim=1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.args.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class QNetwork(nn.Module):
    """Model for DQN and Double DQN"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        h1, h2, h3 = 64, 64, 16
        
        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class DuelingQNetwork(nn.Module):
    """Model for Dueling DQN"""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        h1, h2, h3 = 32, 64, 64
        hv, ha = 128, 128
        
        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        
        
        self.fc_v1 = nn.Linear(h3, hv)
        self.fc_a1 = nn.Linear(h3, ha)
        
        self.fc_v2 = nn.Linear(hv, 1)
        self.fc_a2 = nn.Linear(ha, action_size)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        v = F.relu(self.fc_v1(x))
        v = F.relu(self.fc_v2(v))
        
        a = F.relu(self.fc_a1(x))
        a = F.relu(self.fc_a2(a))
        
        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        
        return x
