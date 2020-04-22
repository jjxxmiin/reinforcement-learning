# -*- coding: utf-8 -*-
from environment import GraphicDisplay, Env


class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    # 가치 이터레이션
    # 벨만 "최적" 방정식을 통해 다음 가치 함수 계산
    def value_iteration(self):
        # 다음 iter 에서 사용할 가치함수를 저장하기 위한 버퍼
        next_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]
        # 모든 상태에 대해서 가치함수 업데이트 하고, next_value_table에 저장
        for state in self.env.get_all_states():
            if state == [2, 2]:  # 종료 state
                next_value_table[state[0]][state[1]] = 0.0  # 종료 state에서의 가치함수값 = 0
                continue
            # 현재 상태에서 선택 가능한 모든 행동에 대한 가치 함수를 저장하기 위한 위한 빈 리스트
            value_list = []

            # 현재 상태에서 선택 가능한 모든 행동에 대해 가치함수 값 계산
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value

                value_list.append(value)

            # 모든 행동에 대한 가치함수 값 중에서 최댓값을 다음 가치 함수로 대입
            next_value_table[state[0]][state[1]] = max(value_list)

        # 모든 상태에 대한 가치함수를 한번에 업데이트
        self.value_table = next_value_table

    # 현재 가치 함수로부터 행동을 반환
    def get_action(self, state):
        action_list = []
        max_value = -99999

        # terminal state (목표하는 종료 state = 푸른색 원이 있는 상태로 들어감)
        if state == [2, 2]:
            return []  # 종료 state 에서는 아무런 행동을 하지 않음.

        # 모든 행동에 대해 큐함수 (보상 + (감가율 * 다음 상태 가치함수))를 계산
        # 최대 큐 함수를 가진 행동(복수일 경우 여러 개)을 반환
        for i, action in enumerate(self.env.possible_actions):
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = reward + self.discount_factor * next_value

            if value > max_value:
                action_list.clear()
                max_value = value
                action_list.append(action)

            elif value == max_value:
                action_list.append(action)

        return action_list

    def get_value(self, state):
        # 상태 state에 대한 가치함수 값 리턴
        return self.value_table[state[0]][state[1]]


if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()

