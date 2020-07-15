import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from net import policy_net, value_net
from buffer import trajectory_buffer


class ppo_clip(object):
    def __init__(self, env_id, epoch, learning_rate, gamma, lam, epsilon, capacity, update_iter, model_id=None, update_freq=50):
        super(ppo_clip, self).__init__()
        self.model_id = model_id
        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epoch = epoch
        self.capacity = capacity
        self.update_iter = update_iter
        self.update_freq = update_freq

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, 1)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.buffer = trajectory_buffer(capacity=self.capacity)
        self.count = 0
        self.train_count = 0

    def reset(self):
        self.count = 0
        self.train_count = 0
        self.buffer.clear()

    def train(self):
        obs, next_obs, act, rew, don, val = self.buffer.get()

        obs = torch.FloatTensor(obs)
        next_obs = torch.FloatTensor(next_obs)
        act = torch.LongTensor(act)
        rew = torch.FloatTensor(rew)
        don = torch.FloatTensor(don)
        val = torch.FloatTensor(val)

        old_probs = self.policy_net.forward(obs)
        old_probs = old_probs.gather(1, act).squeeze(1).detach()
        value_loss_buffer = []
        policy_loss_buffer = []
        for _ in range(self.update_iter):
            td_target = rew + self.gamma * self.value_net.forward(next_obs) * (1 - don)
            delta = td_target - self.value_net.forward(obs)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lam * advantage + delta_t[0]
                advantage_lst.append([advantage])

            advantage_lst.reverse()
            advantage = torch.FloatTensor(advantage_lst)

            value = self.value_net.forward(obs)
            #value_loss = (ret - value).pow(2).mean()
            value_loss = F.smooth_l1_loss(td_target.detach(), value)
            value_loss_buffer.append(value_loss.item())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            probs = self.policy_net.forward(obs)
            probs = probs.gather(1, act).squeeze(1)
            ratio = probs / old_probs
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * advantage
            policy_loss = - torch.min(surr1, surr2).mean()
            policy_loss_buffer.append(policy_loss.item())
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

    def load_weight_hyperparam(self, model_path):
        model_ = torch.load(model_path)
        self.policy_net.load_state_dict(model_['policy_weight'])
        self.value_net.load_state_dict(model_['value_weight'])
        hyperparameters = model_['hyperparameters']
        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.lam = hyperparameters['lam']
        self.epsilon = hyperparameters['epsilon']

    def save_weight_hyperparam(self, model_path):
        model_ = {}
        model_['policy_weight'] = self.policy_net.state_dict()
        model_['value_weight'] = self.value_net.state_dict()
        hyperparameters = {}
        hyperparameters['learning_rate'] = self.learning_rate
        hyperparameters['gamma'] = self.gamma
        hyperparameters['lam'] = self.lam
        hyperparameters['epsilon'] = self.epsilon
        model_['hyperparameters'] = hyperparameters
        torch.save(model_, model_path)

    def run(self):
        while True:
            if self.train_count == self.epoch:
                break
            obs = self.env.reset()
            total_reward = 0
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                next_obs, reward, done, _ = self.env.step(action)
                value = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0))).detach().item()
                self.buffer.store(obs, next_obs, action, reward, done, value)
                self.count += 1
                total_reward += reward
                obs = next_obs
                if self.count % self.update_freq == 0:
                    self.train_count += 1
                    self.train()
                    self.buffer.clear()
                    if self.train_count == self.epoch:
                        break
                if done:
                    break

    def eval(self, num=5):
        score_list = []
        for _ in range(num):
            obs = self.env.reset()
            total_reward = 0
            while True:
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                next_obs, reward, done, _ = self.env.step(action)
                value = self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0))).detach().item()
                total_reward += reward
                obs = next_obs
                if done:
                    break
            score_list.append(total_reward)
        return np.mean(score_list)


if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
