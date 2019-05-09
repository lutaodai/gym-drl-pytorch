import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPS(nn.Module):
    def __init__(self, state_size, action_size, args, device):
        super(SPS, self).__init__()
        self.args = args
        self.device = device
        self.hidden_size = args.hidden_size
        assert isinstance(self.hidden_size, tuple)
        dims = (state_size,) + self.hidden_size + (action_size,)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) 
                                      for i in range(len(dims)-1)])
        self.p = 0
        print("Implementing Stochastic Policy Search")
        print(self.linears)
        self.init_weights()
    
    def init_weights(self):
        for w in self.parameters():
            nn.init.xavier_normal_(w, 1e-4)
        
    def forward(self, x):
        """state -> action prob"""
        ind_end = len(self.linears)
        for i, l in enumerate(self.linears):
            if i == ind_end - 1:
                x = F.softmax(l(x), dim=1)
            else:
                x = F.relu(F.dropout(l(x), p=self.p))
        return x
    
    def act(self, x, stochastic=True):
        x = torch.Tensor(x).unsqueeze(0).to(self.device)
        prob = self.forward(x).cpu().detach().numpy().squeeze(0)
        if stochastic:
            return np.random.choice(len(prob), p=prob)
        else:
            return np.argmax(prob)
    
    def update(self, noise_scale):
        for i, l in enumerate(self.linears):
            l.weight.data = l.weight.data + noise_scale * torch.randn_like(l.weight.data, device=self.device)
            
        