import gym
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import argparse
import copy
import torch.optim as optim

from utils.utils import str2bool, str2tuple, plot_scores, future_rewards, normalize
from agent_wrapper import AgentWrapper

PARSER = argparse.ArgumentParser(description=None)
# choosing algorithm
PARSER.add_argument('-a', '--agent', default='sps', help='name of the agent')
PARSER.add_argument('-aw', '--args_wrapper', default=True, type=str2bool, help='if using default hyperparameter')
PARSER.add_argument('-hs', '--hidden_size', default=(), type=str2tuple, help='hidden size for nn')

# output path
PARSER.add_argument('-fdir', '--figure_dir', default="figure", help="directory storing figures")
PARSER.add_argument('-mdir', '--model_dir', default='model', help="directory storing models")
PARSER.add_argument('-f', '--flag', default='sps', type=str, help="saving name")

# figure setting
PARSER.add_argument('-ws', '--window_size', default=100, type=int,
                    help="moving average window for plotting")

# model hyperparameters
PARSER.add_argument('-gamma', '--gamma', default=1., type=float, help='discount factor')

# training hyperparameters
PARSER.add_argument('-s', '--seed', default=432, type=int, help="seeds")
PARSER.add_argument('-maxt', '--maxt', default=10000, type=int, help="maximum episode length")
PARSER.add_argument('-n', '--n_episode', default=2000, type=int, help="maximum numbers of episode")
PARSER.add_argument('-ns0', '--init_noise_scale', default=1e-2, type=float, help="initial noise scale")
PARSER.add_argument('-ns_min', '--noise_scale_min', default=1e-3, type=float, help="minimum noise scale")
PARSER.add_argument('-ns_max', '--noise_scale_max', default=2, type=float, help="maximum noise scale")
PARSER.add_argument('-ns_gamma', '--noise_scale_gamma', default=2, type=float, help="noise scale variation factor")


ARGS = PARSER.parse_args()

if not os.path.exists(ARGS.figure_dir): os.makedirs(ARGS.figure_dir)
if not os.path.exists(ARGS.model_dir): os.makedirs(ARGS.model_dir)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # making environment
    env = gym.make("CartPole-v1")
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
                                device=device).to(device)
    best_model = copy.deepcopy(agent)

    
    n_episodes = ARGS.n_episode
    max_t = ARGS.maxt
    gamma = ARGS.gamma
    noise_scale = ARGS.init_noise_scale

    scores = []                        
    scores_window = deque(maxlen=100)
    best_R = -np.Inf
    
    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_window.append(sum(rewards))
        scores.append(sum(rewards))
        
        discount = [gamma**i for i in range(len(rewards))]
        R = sum([a*b for a, b in zip(discount, rewards)])
        
        if R >= best_R:
            best_R = R
            best_model = copy.deepcopy(agent)
            noise_scale = max(ARGS.noise_scale_min, noise_scale / ARGS.noise_scale_gamma)
            agent.update(noise_scale)
        else:
            noise_scale = min(ARGS.noise_scale_max, noise_scale * ARGS.noise_scale_gamma)
            agent = copy.deepcopy(best_model)
            agent.update(noise_scale)

        
        print('\rEpisode {}\tAverage Score: {:.2f}\tBest R: {:.2f}'\
              .format(i_episode, np.mean(scores_window), best_R), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'\
                  .format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= env.spec.reward_threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.state_dict(), os.path.join(ARGS.model_dir, 'model_%s.pth' %ARGS.flag))
            break
    
    # generate score plot
    plot_scores(scores, name = ARGS.flag, 
                window_size = ARGS.window_size,
                save_dir = ARGS.figure_dir)