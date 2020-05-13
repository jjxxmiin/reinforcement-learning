import numpy as np
import random
from env import Env
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = actions
        self.learning_rate = 0.01  # 학습률, 2)번 문제
        self.discount_factor = 0.9  # 감가율, 3)번 문제
        self.epsilon = 0.05  # 랜덤 행동을 할 확률
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * max(self.q_table[next_state]) - self.q_table[state][action])

    # 입실론 탐욕 정책에 따라서 행동을 선택
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state_action = self.q_table[state]
            action = self.arg_max(state_action)

        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


if __name__ == "__main__":
    env = Env()  # 환경 객체를 생성
    agent = QLearningAgent(actions=list(range(env.n_actions)))  # Q러닝 Agent 객체 생성

    EPISODE_MAX = 1000
    for episode in range(EPISODE_MAX):
        state = env.reset()  # 환경을 초기화 하고, 초기 상태 s 를 얻기.

        while True:  # 현재 episode가 끝날 때 까지 반복
            env.render()

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(str(state))
            # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴
            next_state, reward, done = env.step(action)

            # <s,a,r,s'> 샘플로 Q 함수를 업데이트
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            if done:  # 현재 에피소드가 끝난경우...
                break  # while-loop를 탈출

