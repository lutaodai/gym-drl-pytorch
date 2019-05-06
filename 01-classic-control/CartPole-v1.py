import gym
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import argparse

from utils.replay_buffer import ReplayBuffer
from utils.utils import str2bool, str2tuple, plot_scores
from agents.dqn import DQNAgent
from agent_wrapper import AgentWrapper

PARSER = argparse.ArgumentParser(description=None)
# choosing algorithm
PARSER.add_argument('-a', '--agent', default='double_dqn', help='name of the agent')
PARSER.add_argument('-aw', '--args_wrapper', default=True, type=str2bool, help='if using default hyperparameter')
PARSER.add_argument('-hs', '--hidden_size', default=(32,64,64), type=str2tuple, help='hidden size for nn')

# output path
PARSER.add_argument('-fdir', '--figure_dir', default="figure", help="directory storing figures")
PARSER.add_argument('-mdir', '--model_dir', default='model', help="directory storing models")
PARSER.add_argument('-f', '--flag', default='double_dqn', type=str, help="saving name")

# figure setting
PARSER.add_argument('-ws', '--window_size', default=100, type=int,
                    help="moving average window for plotting")
# replay buffer
PARSER.add_argument('-bus', '--buffer_size', default=int(1e5), type=int,
                    help='buffer size for experience replay buffer')
# RL hyperparameters
PARSER.add_argument('-gamma', '--gamma', default=1., type=float, help='discount factor')
PARSER.add_argument('-tau', '--tau', default=1e-3, type=float, help='factor for soft update of target parameters')
PARSER.add_argument('-lr', '--lr', default=5e-4, type=float, help='learning rate')
PARSER.add_argument('-uf', '--update_frequency', default=1, type=int, help='how often to update the network')

# training hyperparameters
PARSER.add_argument('-bas', '--batch_size', default=256, type=int,help='batch size training')
PARSER.add_argument('-s', '--seed', default=1, type=int, help="seeds")
PARSER.add_argument('-maxt', '--maxt', default=10000, type=int, help="maximum episode length")
PARSER.add_argument('-n', '--n_episode', default=2000, type=int, help="maximum numbers of episode")
PARSER.add_argument('-e0', '--epi_start', default=0.5, type=float, help='initial epsilon')
PARSER.add_argument('-eT', '--epi_end', default=0.005, type=float, help='minimum epsilon')
PARSER.add_argument('-e_decay', '--epi_decay', default=0.99, type=float, help='epsilon decay rate')
PARSER.add_argument('-b', '--baseline', default=195, type=float, help='stopping baseline')

ARGS = PARSER.parse_args()

if not os.path.exists(ARGS.figure_dir): os.makedirs(ARGS.figure_dir)
if not os.path.exists(ARGS.model_dir): os.makedirs(ARGS.model_dir)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # making environment
    env = gym.make("CartPole-v1")
    env = env.unwrapped
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    
    agent_wrapper = AgentWrapper(ARGS)
    agent_wrapper.wrap()
    if ARGS.args_wrapper:
        ARGS = agent_wrapper.args
    print(ARGS)
    
    # initializing agent
    agent = agent_wrapper.agent(state_size = state_size,
                     action_size = action_size,
                     args=ARGS,
                     device=device)
    
    n_episodes = ARGS.n_episode
    max_t = ARGS.maxt
    eps_start = ARGS.epi_start
    eps_end = ARGS.epi_end
    eps_decay = ARGS.epi_decay
    
    scores = []                        
    scores_window = deque(maxlen=100)  
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            # env.render()
            action = agent.act(state, eps)                 
            next_state, reward, done, info = env.step(action)       
            agent.step(state, action, reward, next_state, done)
            score += reward                                
            state = next_state                             
            if done:                                       
                break

        scores_window.append(score)       
        scores.append(score)              
        eps = max(eps_end, eps_decay*eps) 

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= ARGS.baseline:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(ARGS.model_dir, 'model_%s.pth' %ARGS.flag))
            break
    
    # generate score plot
    plot_scores(scores, name = ARGS.flag, 
                window_size = ARGS.window_size,
                save_dir = ARGS.figure_dir)