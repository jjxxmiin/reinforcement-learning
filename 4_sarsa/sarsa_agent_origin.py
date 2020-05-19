import numpy as np
import pickle
import random
from collections import defaultdict
from env import Env

MAX_EPISODE = 1000

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1  # 3) 시간이 지날수록 e 값이 감소하도록 코드를 수정하세요.
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # 큐함수 업데이트
    def learn(self, s, a, r, s_, a_):
        self.q_table[s][a] += self.learning_rate * (r + self.discount_factor * self.q_table[s_][a_] - self.q_table[s][a])

    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택 (exploration)
            best_action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 최적 행동 반환 (exploitation)
            state_action = self.q_table[state]
            best_action = self.arg_max(state_action)

        return best_action

    """
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
    """

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = -9999

        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)

        return random.choice(max_index_list)


if __name__ == "__main__":
    env = Env()  # 환경에 대한 instance 생성
    agent = SARSAgent(actions=list(range(env.n_actions)))  # Sarsa Agent 객체 생성

    success_total_step = 0
    fail_total_step = 0
    num_success = 0
    num_fail = 0
    step_log = []

    # 지정된 횟수(MAX_EPISODE)만큼 episode 진행
    for episode in range(MAX_EPISODE):
        # 게임 환경과 상태를 초기화 하고, 상태(state)값 얻기
        num_step = 0

        state = env.reset()

        # 현재 상태에서 어떤 행동을 할지 선택
        action = agent.get_action(str(state))

        # 한개의 episode를 처음부터 끝까지 처리하는 while-loop
        while True:
            env.render()

            num_step += 1

            next_state, reward, done = env.step(action)
            next_action = agent.get_action(str(next_state))

            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action
            # action = agent.get_action(str(next_state))

            # 모든 큐함수 값을 화면에 표시
            env.print_value_all(agent.q_table)

            # episode가 끝났으면 while-loop을 종료
            if done:
                if reward > 0:
                    num_success += 1
                    print("success")
                    success_total_step += num_step

                else:
                    num_fail += 1
                    print("fail")
                    fail_total_step += num_step

                step_log.append(num_step)
                break

    with open('pkl/log.pkl', 'wb') as f:
        pickle.dump([step_log,
                     num_success,
                     num_fail,
                     success_total_step,
                     fail_total_step], f)