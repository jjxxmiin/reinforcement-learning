import copy

import numpy as np
import pylab
from env import Env
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 2500


class ReinForceAgent:
    def __init__(self, pretrain=False):
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)

        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()

        self.states = []
        self.actions = []
        self.rewards = []

        if pretrain:
            self.model.load_weights('save_model/REINFORCE.h5')

    def get_action(self, state):
        output = self.model.predict(state)

        return np.random.choice(self.action_size, 1, p=output[0])

    def append_sample(self, state, action, reward):
        one_hot_action = np.zeros_like(self.action_space)
        one_hot_action[action] = 1

        self.states.append(state[0])
        self.actions.append(one_hot_action)
        self.rewards.append(reward)

    def discount_rewards(self, rewards):
        G = []
        temp = 0

        for r in rewards[::-1]:
            temp = temp * self.discount_factor + r
            G.append(temp)

        return G[::-1]

    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def build_optimizer(self):
        action = K.placeholder(shape=[None, 5])
        discounted_rewards = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)

        return train

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model


if __name__ == "__main__":
    env = Env()
    agent = ReinForceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while True:
            global_step += 1
            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.append_sample(state, action, reward)

            state = copy.deepcopy(next_state)

            score += reward

            if done:
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("save_graph/REINFORCE.png")
                score = round(score, 2)
                print(f"episode: {e} score: {score} time_step: {global_step}")
                break

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights("save_model/REINFORCE.h5")



