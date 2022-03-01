import math
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import torch
from cost.pedestrian_proximity_to_block_cost import PedestrianProximityToBlockCost
from cost.proximity_to_block_cost import ProximityToDownBlockCost, ProximityToLeftBlockCost, ProximityToUpBlockCost
from player_cost.player_cost import PlayerCost

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

def draw_real_car(player_id, car_states, path=None, alpha=1.0):
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
            path = "visual_components/delorean-flux-white.png" if path is None else path
        else:
            state = car_states[i][5:].flatten()
            color = "g"
            path = "visual_components/car_robot_r.png" if path is None else path

        transform_data = Affine2D().rotate_deg_around(*(state[0], state[1]), state[2]/np.pi * 180) + plt.gca().transData
        # plt.plot(state[0], state[1], color=color, marker='o', markersize=5, alpha = 0.4)
        if i % 5 == 0:
            plt.imshow(
                plt.imread(path, format="png"), 
                transform = transform_data, 
                interpolation='none',
                origin='lower',
                extent=[state[0] - 0.927, state[0] + 3.34, state[1] - 0.944, state[1] + 1.044],
                alpha = alpha, 
                # alpha=(1.0/len(car_states))*i,
                zorder = 10.0,
                clip_on=True)

def draw_real_human(states, variation=0, alpha=1.0):
    for i in range(len(states)):
        state = states[i][10:]
        transform_data = Affine2D().rotate_deg_around(*(state[0], state[1]), (state[2] + np.pi * 0.5)/np.pi * 180) + plt.gca().transData
        plt.imshow(
            plt.imread("visual_components/human-walking-topdown-step{}.png".format(variation), format="png"), 
            transform = transform_data, 
            interpolation='none',
            origin='lower',
            extent=[state[0] - 1.2, state[0] + 1.2, state[1] + 1.2, state[1] - 1.2],
            zorder = 15.0,
            clip_on=True,
            alpha=alpha
        )

def draw_crosswalk(x, y, width, length, number_of_dashes = 5, border_color="white"):
    per_length = length * 0.5 / number_of_dashes
    for i in range(number_of_dashes):
        crosswalk = plt.Rectangle(
            [x + (2*i + 0.5)*per_length, y], width = per_length, height = width, color = border_color, lw = 0, zorder = 0)
        plt.gca().add_patch(crosswalk)

def plot_road_game(ped=False, adversarial=False, boundary_only=False):
        # Create game env
    ###################
    road_rules = {
        "x_min": 2,
        "x_max": 9.4,
        "y_max": 27.4,
        "y_min": 20,
        "width": 3.7
    }

    ped_road_rules = {
        "x_min": 2,
        "x_max": 9.4,
        "y_max": 31,
        "y_min": 29
    }

    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }

    collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

    g_params = {
        "car1": {
            "position_indices": [(0,1), (5, 6), (10, 11)],
            "player_id": 0, 
            "road_logic": [0, 1, 0, 1, 0],
            "road_rules": road_rules,
            "collision_r": collision_r,
            "goals": [20, 35],
            "car_params": car_params,
            "theta_indices": [2, 7, 12]
        },
        "car2": {
            "position_indices": [(0,1), (5, 6), (10, 11)],
            "player_id": 1, 
            "road_logic": [1, 0, 0, 1, 0],
            "road_rules": road_rules,
            "collision_r": collision_r,
            "goals": [20, 0],
            "car_params": car_params,
            "theta_indices": [2, 7, 12]
        }
    }
    if ped:
        g_params["ped1"] = {
            "position_indices": [(0,1), (5, 6), (10, 11)],
            "player_id": 2, 
            "road_logic": [1, 1, 1, 1, 0],
            "road_rules": ped_road_rules,
            "goals": [15, 30],
            "car_params": car_params,
            "theta_indices": [2, 7, 12]
        }
    ###################
    # Create environment:
    car1_goal_cost = ProximityToUpBlockCost(g_params["car1"])
    if not adversarial:
        car2_goal_cost = ProximityToLeftBlockCost(g_params["car2"])
    else:
        car2_goal_cost = ProximityToDownBlockCost(g_params["car2"])
    if ped:
        ped_goal_cost = PedestrianProximityToBlockCost(g_params["ped1"])

    # Build up total costs for both players. This is basically a zero-sum game.
    car1_cost = PlayerCost()
    car1_cost.add_cost(car1_goal_cost, "x", 1.0)

    car2_cost = PlayerCost()
    car2_cost.add_cost(car2_goal_cost, "x", 1.0)

    if ped:
        ped_cost = PlayerCost()
        ped_cost.add_cost(ped_goal_cost, "x", 1.0)
        
    _renderable_costs = [car1_goal_cost, car2_goal_cost]
    if ped:
        _renderable_costs.append(ped_goal_cost)

    plt.figure(0)
    _plot_lims = [-5, 25, 0, 40]

    ratio = (_plot_lims[1] - _plot_lims[0])/(_plot_lims[3] - _plot_lims[2])
    plt.gcf().set_size_inches(ratio*10.0, 10.0)

    ax = plt.gca()
    plt.axis("off")

    if _plot_lims is not None:
        ax.set_xlim(_plot_lims[0], _plot_lims[1])
        ax.set_ylim(_plot_lims[2], _plot_lims[3])

    ax.set_aspect("equal")

    # Render all costs.
    for cost in _renderable_costs:
        cost.render(ax)
    
    x_max = 25
    y_max = 40

    if not boundary_only:
        grass = plt.Rectangle(
            [-5, 0], width = 30, height = 40, color = "k", lw = 0, zorder = -2, alpha = 0.5)
        plt.gca().add_patch(grass)

    # plot road rules
    x_center = road_rules["x_min"] + 0.5 * (road_rules["x_max"] - road_rules["x_min"])
    y_center = road_rules["y_min"] + 0.5 * (road_rules["y_max"] - road_rules["y_min"])

    if not boundary_only:
        road = plt.Rectangle(
            [road_rules["x_min"], 0], width = road_rules["x_max"] - road_rules["x_min"], height = y_max, color = "darkgray", lw = 0, zorder = -2)
        plt.gca().add_patch(road)
        road = plt.Rectangle(
            [road_rules["x_max"], road_rules["y_min"]], width = x_max, height = road_rules["y_max"] - road_rules["y_min"], color = "darkgray", lw = 0, zorder = -2)
        plt.gca().add_patch(road)

    if boundary_only:
        border_color = "black"
    else:
        border_color = "white"

    crosswalk_width = 3
    crosswalk_length = road_rules["x_max"] - road_rules["x_min"]
    draw_crosswalk(road_rules["x_min"], 30 - crosswalk_width*0.5, crosswalk_width, crosswalk_length, border_color=border_color)

    ax.plot([road_rules["x_min"], road_rules["x_min"]], [0, y_max], c=border_color, linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], road_rules["x_max"]], [0, road_rules["y_min"]], c=border_color, linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], road_rules["x_max"]], [road_rules["y_max"], y_max], c=border_color, linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_min"], road_rules["y_min"]], c=border_color, linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [road_rules["y_min"], road_rules["y_min"]], c=border_color, linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_max"], road_rules["y_max"]], c=border_color, linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [road_rules["y_max"], road_rules["y_max"]], c=border_color, linewidth = 2, zorder = -1)
    ax.plot([x_center, x_center], [0, y_max], "--", c = 'white', linewidth = 5, dashes=(5, 5), zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [y_center, y_center], "--", c = 'white', linewidth = 5, dashes=(5, 5), zorder = -1)