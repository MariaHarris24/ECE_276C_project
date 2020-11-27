# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:13:55 2020

@author: Call of Duty
"""
import random


import pickle


import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import gym
import pybullet
import pybulletgym.envs
import IPython
from IPython import display
import pybullet_envs

import matplotlib.pyplot as plt
np.random.seed(50)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Walker2DMuJoCoEnv-v0")
env.render()
state = env.reset()
done = False
while not done:
    # env.env._cam_dist=1
    # env.env._cam_pitch=-90
    
    plt.imshow(env.render(mode='rgb_array'))
    #display.clear_output(wait=True)
    #display.display(plt.gcf)
    plt.show()
    env.render()

    action = env.action_space.sample()
    next_state, r, done, _ = env.step(action)
    state = next_state
    
print(env.observation_space.shape)
print(env.action_space.shape)
print(env.action_space.high[0])