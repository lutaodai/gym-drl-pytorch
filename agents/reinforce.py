import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Reinforce:
    def __init__(self, state_size, action_size, args, device):
        self.policy = Policy(state_size, action_size, args, device).to(device)
        print("Implmenting REINFORCE")
        print(self.policy)

    
class Policy(nn.Module):
    def __init__(self, state_size, action_size, args, device):
        super(Policy, self).__init__()
        torch.manual_seed(args.seed)
        self.device = device
        assert isinstance(args.hidden_size, tuple)
        
        dims = (state_size,) + args.hidden_size + (action_size,)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) 
                                      for i in range(len(dims)-1)])
        self.p = args.dropout
        self.init_weights()
        
    def init_weights(self):
        for w in self.parameters():
            nn.init.xavier_normal_(w, 1e-4)
        
    def forward(self, x):
        ind_end = len(self.linears)
        for i, l in enumerate(self.linears):
            if i == ind_end - 1:
                x = F.softmax(l(x), dim=1)
            else:
                x = F.relu(F.dropout(l(x), p=self.p))
        return x
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    
    