from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import torch

class MaxFuncMux(object):
    def __init__(self):
        self.io = dict()
        self.io_torch = dict()
        self.torch = False

    def store(self, func, func_val):
        self.io[func] = func_val
        if type(func_val) is torch.Tensor:
            self.torch = True
            self.io_torch[func] = func_val
    
    def get_max(self):
        """
        return the max value, and the function that gives out that max value
        return func_of_max_val, max_val
        """
        if self.torch:
            key = max(self.io, key = self.io.get)
            return key, self.io_torch[key]
        else:
            return max(self.io, key = self.io.get), max(self.io.values())

class MinFuncMux(object):
    def __init__(self):
        self.io = dict()
        self.io_torch = dict()
        self.torch = False

    def store(self, func, func_val):
        self.io[func] = func_val
        if type(func_val) is torch.Tensor:
            self.torch = True
            self.io_torch[func] = func_val
    
    def get_min(self):
        """
        return the min value, and the function that gives out that min value
        return func_of_min_val, min_val
        """
        if self.torch:
            key = min(self.io, key = self.io.get)
            return key, self.io_torch[key]
        else:
            return min(self.io, key = self.io.get), min(self.io.values())

def draw_real_car(player_id, car_states):
    # TODO: change all the constants in the function to car_params
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }
    
    for i in range(len(car_states)):
        if player_id == 0:
            state = car_states[i][:5].flatten()
            color = "r"
            path = "visual_components/delorean.png"
        else:
            state = car_states[i][5:].flatten()
            color = "g"
            path = "visual_components/car_robot_r.png"

        transform_data = Affine2D().rotate_deg_around(*(state[0], state[1]), state[2]/np.pi * 180) + plt.gca().transData
        # plt.plot(state[0], state[1], color=color, marker='o', markersize=5, alpha = 0.4)
        if i % 5 == 0:
            plt.imshow(
                plt.imread(path, format="png"), 
                transform = transform_data, 
                interpolation='none',
                origin='lower',
                extent=[state[0] - 0.927, state[0] + 3.34, state[1] - 0.944, state[1] + 1.044],
                alpha = 1.0, 
                # alpha=(1.0/len(car_states))*i,
                zorder = 10.0,
                clip_on=True)

def draw_real_human(states, variation=0):
    for i in range(len(states)):
        state = states[i][10:]
        transform_data = Affine2D().rotate_deg_around(*(state[0], state[1]), (state[2] + np.pi * 0.5)/np.pi * 180) + plt.gca().transData
        plt.imshow(
            plt.imread("visual_components/human-walking-topdown-step{}.png".format(variation), format="png"), 
            transform = transform_data, 
            interpolation='none',
            origin='lower',
            extent=[state[0] - 1.2, state[0] + 1.2, state[1] + 1.2, state[1] - 1.2],
            zorder = 10.0,
            clip_on=True
        )

def draw_crosswalk(x, y, width, length, number_of_dashes = 5):
    per_length = length * 0.5 / number_of_dashes
    for i in range(number_of_dashes):
        crosswalk = plt.Rectangle(
            [x + (2*i + 0.5)*per_length, y], width = per_length, height = width, color = "white", lw = 0, zorder = 0)
        plt.gca().add_patch(crosswalk)