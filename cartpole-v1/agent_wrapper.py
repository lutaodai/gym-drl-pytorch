from agents.dqn import DQNAgent
from agents.reinforce import Reinforce

class AgentWrapper:
    def __init__(self, args):
        agent_list = ['dqn', 'double_dqn', 'dueling_dqn', 'reinforce']
        self.args = args
        self.agent_name = args.agent
        assert self.agent_name in agent_list, "invalid agent name. agent name \
                                          should be one of the " + str(agent_list)
        
        
    def wrap(self):
        eval('self.%s()' %self.agent_name)
    
    def dqn(self):
        if self.args.args_wrapper:
            self.args.flag = 'dqn'
            self.args.hidden_size = (64,128,256)
            self.args.window_size = 100
            self.args.buffer_size = int(1e5)
            self.args.gamma = 1.
            self.args.tau = 1e-3
            self.args.lr = 1e-3
            self.args.update_frequency = 1
            self.args.batch_size = 64
            self.args.maxt = 10000
            self.args.n_episode = 2000
            self.args.epi_start = 1.
            self.args.epi_end = 0.01
            self.args.epi_decay = 0.995
        self.agent = DQNAgent
    
    def double_dqn(self):
        if self.args.args_wrapper:
            self.args.flag = 'double_dqn'
            self.args.hidden_size = (64,128,256)
            self.args.window_size = 100
            self.args.buffer_size = int(1e5)
            self.args.gamma = 1.
            self.args.tau = 1e-3
            self.args.lr = 1e-3
            self.args.update_frequency = 1
            self.args.batch_size = 64
            self.args.maxt = 10000
            self.args.n_episode = 2000
            self.args.epi_start = 1.
            self.args.epi_end = 0.01
            self.args.epi_decay = 0.995
        self.agent = DQNAgent
    
    def dueling_dqn(self):
        if self.args.args_wrapper:
            self.args.flag = 'dueling_dqn'
            # self.args.hidden_size = ((32,64), (128,), (64, 128)) #base, value, advantage
            self.args.hidden_size = ((32,64,64), (64, 64), (64,128)) #base, value, advantage
            self.args.window_size = 100
            self.args.buffer_size = int(1e5)
            self.args.gamma = 1.
            self.args.tau = 1e-3
            self.args.lr = 5e-4
            self.args.update_frequency = 1
            self.args.batch_size = 64
            self.args.maxt = 10000
            self.args.n_episode = 2000
            self.args.epi_start = 1.
            self.args.epi_end = 0.01
            self.args.epi_decay = 0.995
        self.agent = DQNAgent
    
    def reinforce(self):
        if self.args.args_wrapper:
            self.args.flag = 'reinforce'
            self.args.hidden_size = (128,)
            self.args.window_size = 100
            self.args.gamma = 0.99
            self.args.lr = 1e-2
            self.args.maxt = 10000
            self.args.n_episode = 2000
        self.agent = Reinforce