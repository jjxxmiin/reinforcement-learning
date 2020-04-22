# -*- coding: utf-8 -*-
import random
from environment import GraphicDisplay, Env


class PolicyIteration:
    def __init__(self, env):
        # 환경 객체 저장
        self.env = env
        # 가치함수를 2차원 리스트로 초기화 (상태별 가치함수 값 저장)
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # 초기 정책 : 랜덤(상 하 좌 우 동일한 확률로 정책 초기화)
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                             for _ in range(env.height)]
        # 목표하는 종료 상태(=목적지에 도달한 경우)의 정책 = 정지(이동하지 않음)
        self.policy_table[2][2] = []
        # 감가율
        self.discount_factor = 0.9

    def policy_evaluation(self):
        # 다음 가치함수를 저장할 테이블/버퍼 초기화
        # self.value_table 은 마지막에 한번에 업데이트 해야함
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        # 모든 상태에 대해서 벨만 기대 방정식 계산
        for state in self.env.get_all_states():
            value = 0.0

            # 목표하는 종료 상태의 가치 함수 = 0 으로 설정
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            # 벨만 기대 방정식
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] *
                          (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)  # 반올림 (x.xx 까지)

        # 가치함수 테이블을 다음 가치함수로 업데이트.
        # 이렇게 해야 모든 상태에 대한 가치함수를 한번에 업데이트 할 수 있음
        self.value_table = next_value_table

    def policy_improvement(self):  # 현재 가치 함수에 대해서 탐욕 정책 발전
        # 현재 policy 를 저장할 테이블/버퍼 생성
        # self.policy_table 은 마지막에 한번에 업데이트 해야함
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:  # goal state
                continue
            value = -99999  # 현재 상태에서의 가장 큰 q값을 저장할 변수
            max_index = []  # 현재 상테에서 가장 큰 q값을 반환하는 행동에 대한 인덱스를 저장
            # 반환할 정책 초기화 (현재 상태에서의 업데이트 된 정책을 저장할 버퍼)
            result = [0.0, 0.0, 0.0, 0.0]

            # 모든 "행동"에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
                # 탐욕 정책 발전
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # 행동의 확률 계산
            # 최고의 행동이 여러개인 경우, 해당 행동들에 대해 동일한 확률 부여
            prob = 1 / len(max_index)

            for index in max_index:
                result[index] = prob

            # 상태 s에 대한 정책 업데이트
            next_policy[state[0]][state[1]] = result

        # 정책 업데이트 : 모든 상태에 대한 정책을 한번에 업데이트 하기 위해 next_policy 라는 버퍼를 사용함
        self.policy_table = next_policy

    # 특정 상태에서 정책에 따른 행동을 반환
    def get_action(self, state):
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # 정책에 담긴 행동 중에 무작위로 한 행동을 추출
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    # 가치 함수의 값을 반환
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]], 2)


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
