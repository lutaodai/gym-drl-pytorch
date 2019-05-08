import gym
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import argparse
import torch.optim as optim

from utils.utils import str2bool, str2tuple, plot_scores, future_rewards, normalize
from agent_wrapper import AgentWrapper

PARSER = argparse.ArgumentParser(description=None)
# choosing algorithm
PARSER.add_argument('-a', '--agent', default='reinforce', help='name of the agent')
PARSER.add_argument('-aw', '--args_wrapper', default=True, type=str2bool, help='if using default hyperparameter')
PARSER.add_argument('-hs', '--hidden_size', default=(32,64,64), type=str2tuple, help='hidden size for nn')

# output path
PARSER.add_argument('-fdir', '--figure_dir', default="figure", help="directory storing figures")
PARSER.add_argument('-mdir', '--model_dir', default='model', help="directory storing models")
PARSER.add_argument('-f', '--flag', default='reinforce', type=str, help="saving name")

# figure setting
PARSER.add_argument('-ws', '--window_size', default=100, type=int,
                    help="moving average window for plotting")

# model hyperparameters
PARSER.add_argument('-gamma', '--gamma', default=1., type=float, help='discount factor')
PARSER.add_argument('-lr', '--lr', default=5e-4, type=float, help='learning rate')
PARSER.add_argument('-d', '--dropout', default=0.5, type=float, help='dropout probability')

# training hyperparameters
PARSER.add_argument('-s', '--seed', default=432, type=int, help="seeds")
PARSER.add_argument('-maxt', '--maxt', default=10000, type=int, help="maximum episode length")
PARSER.add_argument('-n', '--n_episode', default=2000, type=int, help="maximum numbers of episode")

ARGS = PARSER.parse_args()

if not os.path.exists(ARGS.figure_dir): os.makedirs(ARGS.figure_dir)
if not os.path.exists(ARGS.model_dir): os.makedirs(ARGS.model_dir)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # making environment
    env = gym.make("CartPole-v1")
    # env = env.unwrapped
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
    gamma = ARGS.gamma
    optimizer = optim.Adam(agent.policy.parameters(), lr=ARGS.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, 
                                                     verbose=True, threshold=1, 
                                                     min_lr = 5e-4, patience=100)

    scores = []                        
    scores_window = deque(maxlen=100)  
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = torch.Tensor([]).to(device)
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = agent.policy.act(state)
            saved_log_probs = torch.cat([saved_log_probs, log_prob.reshape(1)])
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_window.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Convert rewards to future rewards and normalize them
        R = future_rewards(rewards, gamma)
        R = torch.FloatTensor(R).squeeze(1).to(device)
        R = normalize(R)

        policy_loss = (torch.sum(torch.mul(saved_log_probs, R).mul(-1), -1))
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        scheduler.step(np.mean(scores_window))
        
        print('\rEpisode {}\tAverage Score: {:.2f}'\
              .format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'\
                  .format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= env.spec.reward_threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.policy.state_dict(), os.path.join(ARGS.model_dir, 'model_%s.pth' %ARGS.flag))
            break
    
    # generate score plot
    plot_scores(scores, name = ARGS.flag, 
                window_size = ARGS.window_size,
                save_dir = ARGS.figure_dir)