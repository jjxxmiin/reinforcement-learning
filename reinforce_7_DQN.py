# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:57:45 2019

@author: JM
"""

import numpy as np
import tensorflow as tf
import random
#import dqn
from collections import deque

import gym
env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

print(input_size,output_size)

dis = 0.9
REPLAY_MEMORY = 50000

