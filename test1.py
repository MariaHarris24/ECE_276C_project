# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:13:55 2020

@author: Call of Duty
"""
import copy
import random
import time

import pickle

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import gym
import pybullet
import pybulletgym.envs
import pybullet_envs

import IPython
from IPython import display

import matplotlib.pyplot as plt

RANDOM_SEED = 50
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Using device:', torch.cuda.get_device_name(0))
else:
    print('Using device:', device)
    
e = "Walker2DMuJoCoEnv-v0"
env = gym.make(e)
env.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
env.render()
state = env.reset()
done = False
step = 0

plt.figure()
t = 'seed = {}, step {}'.format(RANDOM_SEED, step)
plt.imshow(env.render(mode='rgb_array'))
#display.clear_output(wait=True)
#display.display(plt.gcf)
plt.title(t)
plt.savefig(t.replace(':', ' - '), dpi=400)
plt.show()
while not done:
    # env.env._cam_dist=1
    # env.env._cam_pitch=-90
    action = ddpg_object.actor(torch.FloatTensor(state).to(device)).detach().cpu().squeeze().numpy()
    next_state, r, done, _ = env.step(action)
    state = next_state
    step += 1
    
    plt.figure()
    t = 'DDPG seed = {}, step {}'.format(RANDOM_SEED, step)
    plt.imshow(env.render(mode='rgb_array'))
    #display.clear_output(wait=True)
    #display.display(plt.gcf)
    plt.title(t)
    plt.savefig(t.replace(':', ' - ') + ".png", dpi=400)
    plt.show()

    
print(env.observation_space.shape)
print(env.action_space.shape)
print(env.action_space.high[0])