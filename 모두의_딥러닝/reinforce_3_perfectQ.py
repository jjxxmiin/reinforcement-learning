# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 22:27:34 2019

@author: JM
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
# from gym.envs.registration import register

env = gym.make('FrozenLake-v2')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
dis = .99
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    e = 1. / ((i//100)+1)

    # The Q-Table learning algorithm
    while not done:
        # choose an action by greedily (with noise) picking from Q table
        # action = np.argmax(Q[state, :] + np.random.randn(1,env.action_space.n) / (i+1))
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state,:])
        
        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="red")
plt.show()

