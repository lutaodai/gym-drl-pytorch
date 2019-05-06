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
        self.hidden_size = args.hidden_size
        self.seed = args.seed
        self.args = args
        self.device = device
        assert self.args.agent in ['dqn', 'double_dqn', 'dueling_dqn'],\
                "invalid agent name"
        if self.args.agent == "double_dqn":
            print("Implementing Double DQN!")
        elif self.args.agent == "dueling_dqn":
            print("Implementing Dueling DQN!")
        else:
            print("Implementing DQN")

        # Q-Network
        if self.args.agent == "dueling_dqn":
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, self.hidden_size, self.seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, self.hidden_size, self.seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, self.hidden_size, self.seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, self.hidden_size, self.seed).to(device)
        print("Agent Architecture")
        print(self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.args.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, args.buffer_size, args.batch_size, self.seed, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = args.update_frequency
    
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
        if self.args.agent == "double_dqn":
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

    def __init__(self, state_size, action_size, hidden_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        dims = (state_size,) + tuple(hidden_size) + (action_size,)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])


    def forward(self, x):
        """Build a network that maps state -> action values."""
        ind_end = len(self.linears)
        for i, l in enumerate(self.linears):
            if i == ind_end:
                x = l(x)
            else:
                x = F.relu(l(x))
        return x
    
class DuelingQNetwork(nn.Module):
    """Model for Dueling DQN"""
    
    def __init__(self, state_size, action_size, hidden_size, seed):
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
        assert len(hidden_size) == 3
        assert isinstance(hidden_size[0], tuple)
        assert isinstance(hidden_size[1], tuple)
        assert isinstance(hidden_size[2], tuple)
        
        base_dims = (state_size,) + hidden_size[0]
        self.base = nn.ModuleList([nn.Linear(base_dims[i], base_dims[i+1])
                                      for i in range(len(base_dims)-1)])
        
        v_dims = (base_dims[-1], ) + hidden_size[1] + (1,)
        self.branch_v = nn.ModuleList([nn.Linear(v_dims[i], v_dims[i+1])
                                      for i in range(len(v_dims)-1)])
        
        a_dims = (base_dims[-1], ) + hidden_size[2] + (action_size, )
        self.branch_a = nn.ModuleList([nn.Linear(a_dims[i], a_dims[i+1])
                                      for i in range(len(a_dims)-1)])
    
    def forward(self, x):
        """Build a network that maps state -> action values."""
        for i, l in enumerate(self.base):
            x = F.relu(l(x))
        v, a = x, x
        for i, l in enumerate(self.branch_v):
            v = F.relu(l(v))
        for i, l in enumerate(self.branch_a):
            a = F.relu(l(a))

        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        
        return x
