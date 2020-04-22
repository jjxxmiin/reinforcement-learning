import tkinter as tk
import time
import numpy as np
import random
from tkinter import Button
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100
HEIGHT = 5
WIDTH = 5
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3]
ACTION = [(-1, 0), (1, 0), (0, -1), (0, 1)]

REWARDS = []


class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Value Iteration')
        self.geometry(f'{HEIGHT * UNIT}x{WIDTH * UNIT + 50}')
        self.texts = []
        self.arrows = []
        self.env = Env()
        self.agent = agent
        self.iteration_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.text_reward(2, 2, "R : 1.0")
        self.text_reward(1, 2, "R : -1.0")
        self.text_reward(2, 1, "R : -1.0")

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)

        calculate_button = Button(self,
                                  text='Calculate',
                                  command=self.calculate_value)
        calculate_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.13, HEIGHT * UNIT + 10,
                             window=calculate_button)

        print_policy_button = Button(self,
                                     text='Print Policy',
                                     command=self.print_optimal_policy)

        print_policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.37, HEIGHT * UNIT + 10,
                             window=print_policy_button)

        move_button = Button(self,
                             text='Move',
                             command=self.move_by_policy)
        move_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.62, HEIGHT * UNIT + 10,
                             window=move_button)

        clear_button = Button(self,
                              text='Clear',
                              command=self.clear)
        clear_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.87, HEIGHT * UNIT + 10,
                             window=clear_button)

        for col in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)

        for row in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, HEIGHT * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])

        canvas.create_image(250, 150, image=self.shapes[1])
        canvas.create_image(150, 250, image=self.shapes[1])
        canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()

        return canvas

    """
    #####################################################################
    ####################### Button action function ######################
    #####################################################################
    """

    def calculate_value(self):
        self.iteration_count += 1

        for i in self.texts:
            self.canvas.delete(i)

        self.agent.value_iteration()
        self.print_values(self.agent.value_table)

    def print_optimal_policy(self):
        self.improvement_count += 1

        for i in self.arrows:
            self.canvas.delete(i)

        for state in self.env.get_all_states():
            action = self.agent.get_action(state)
            self.draw_from_values(state, action)

    def move_by_policy(self):
        if self.improvement_count != 0 and self.is_moving != 1:
            self.is_moving = 1
            x, y = self.canvas.coords(self.rectangle)

            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
            x, y = self.find_rectangle()

            while len(self.agent.get_action([x, y])) != 0:
                action = random.sample(self.agent.get_action([x, y]), 1)[0]

                self.after(100, self.rectangle_move(action))
                x, y = self.find_rectangle()

            self.is_moving = 0

    def clear(self):
        if self.is_moving == 0:
            self.iteration_count = 0
            self.improvement_count = 0

            for i in self.texts:
                self.canvas.delete(i)

            for i in self.arrows:
                self.canvas.delete(i)

            self.agent.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]

            x, y = self.canvas.coords(self.rectangle)
            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

    """
    #####################################################################
    ########################### util function ###########################
    #####################################################################
    """

    def draw_one_arrow(self, col, row, action):
        if col == 2 and row == 2:
            return
        if action == 0:  # up
            origin_x, origin_y = 50 + (UNIT * row), 10 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.up))
        elif action == 1:  # down
            origin_x, origin_y = 50 + (UNIT * row), 90 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.down))
        elif action == 3:  # right
            origin_x, origin_y = 90 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.right))
        elif action == 2:  # left
            origin_x, origin_y = 10 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.left))

    def rectangle_move(self, action):
        base_action = np.array([0, 0])
        location = self.find_rectangle()

        self.render()

        if action == 0 and location[0] > 0:  # up
            base_action[1] -= UNIT
        elif action == 1 and location[0] < HEIGHT - 1:  # down
            base_action[1] += UNIT
        elif action == 2 and location[1] > 0:  # left
            base_action[0] -= UNIT
        elif action == 3 and location[1] < WIDTH - 1:  # right
            base_action[0] += UNIT

        self.canvas.move(self.rectangle, base_action[0], base_action[1])  # move agent

    def find_rectangle(self):
        temp = self.canvas.coords(self.rectangle)
        x = (temp[0] / 100) - 0.5
        y = (temp[1] / 100) - 0.5

        return int(y), int(x)

    def text_value(self, row, col, contents, font='Helvetica', size=12, style='normal', anchor="nw"):
        origin_x, origin_y = 85, 70
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)

        return self.texts.append(text)

    def text_reward(self, row, col, contents, font='Helvetica', size=12, style='normal', anchor="nw"):
        origin_x, origin_y = 5, 5
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)

        return self.texts.append(text)

    def draw_from_values(self, state, action_list):
        i = state[0]
        j = state[1]
        for action in action_list:
            self.draw_one_arrow(i, j, action)

    def print_values(self, values):
        for i in range(WIDTH):
            for j in range(HEIGHT):
                self.text_value(i, j, round(values[i][j], 2))

    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.rectangle)
        self.update()

    """
    #######################################################################
    ########################### static function ###########################
    #######################################################################
    """

    @staticmethod
    def load_images():
        up = PhotoImage(Image.open("img/up.png").resize((13, 13)))
        right = PhotoImage(Image.open("img/right.png").resize((13, 13)))
        left = PhotoImage(Image.open("img/left.png").resize((13, 13)))
        down = PhotoImage(Image.open("img/down.png").resize((13, 13)))
        rectangle = PhotoImage(Image.open("img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("img/circle.png").resize((65, 65)))

        return (up, down, left, right), (rectangle, triangle, circle)


class Env:
    def __init__(self):
        self.transition_probability = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[2][2] = 1
        self.reward[1][2] = -1
        self.reward[2][1] = -1
        self.all_state = []

        for x in range(WIDTH):
            for y in range(HEIGHT):
                state = [x, y]
                self.all_state.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)

        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTION[action_index]

        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1
                    if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1
                    if state[1] > HEIGHT - 1 else state[1])

        return state

    def get_transition_prob(self):
        return self.transition_probability

    def get_all_states(self):
        return self.all_state


if __name__ == '__main__':
    grid_world = GraphicDisplay(None)
    grid_world.mainloop()
