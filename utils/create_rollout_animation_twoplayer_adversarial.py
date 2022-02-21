# create the rollout animation of the three-player game using the log trajectory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation, markers
import os
from matplotlib.transforms import Affine2D
from resource.car_5d import Car5D
from cost.pedestrian_proximity_to_block_cost import PedestrianProximityToBlockCost
from cost.proximity_cost import ProximityToBlockCost
from player_cost.player_cost import PlayerCost
from resource.unicycle_4d import Unicycle4D

import math
import pandas as pd
import imageio

#! NEED TO MERGE WITH evaluate.py

def draw_real_car(player_id, car_states, path=None):
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
            path = "visual_components/delorean.png" if path is None else path
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

# Read log
dir = "logs"
experiment_name = "two_player_time_inconsistent_adversarial"
file_name_leading = "twoplayer_intersection_"
iteration = 350

file_list = os.listdir(os.path.join(dir, experiment_name))
file_index_list = sorted([int(file.replace(file_name_leading, "").replace(".txt","")) for file in file_list])

closest_index = list(iteration - np.array(file_index_list)).index(min(abs(iteration - np.array(file_index_list))))
print("Getting closest file to run iteration: {}: {}".format(iteration, file_index_list[closest_index]))

file_name = os.path.join(dir, experiment_name, file_name_leading + str(file_index_list[closest_index]) + ".txt")
data = pd.read_csv(file_name, sep=",", header=None)
data.columns = [
    "x0", "y0", "theta0", "phi0", "vel0",
    "x1", "y1", "theta1", "phi1", "vel1"
]

# Create game env
# General parameters.
TIME_HORIZON = 3.0
TIME_RESOLUTION = 0.1
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)

car1 = Car5D(2.413)
car2 = Car5D(2.413)

car1_theta0 = np.pi / 2.01
car1_v0 = 5.0
car1_x0 = np.array([
    [7.75],
    [10.0],
    [car1_theta0],
    [0.0],
    [car1_v0]
])

car2_theta0 = -np.pi / 2.01
car2_v0 = 5.0             
car2_x0 = np.array([
    [3.75],
    [35.0],
    [car2_theta0],
    [0.0],
    [car2_v0]
])

###################
road_rules = {
    "x_min": 2,
    "x_max": 9.4,
    "y_max": 27.4,
    "y_min": 20,
    "width": 3.7
}

car_params = {
    "wheelbase": 2.413, 
    "length": 4.267,
    "width": 1.988
}

collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

g_params = {
    "car1": {
        "position_indices": [(0,1), (5, 6)],
        "player_id": 0, 
        "road_logic": [0, 1, 0, 0, 0],
        "road_rules": road_rules,
        "collision_r": collision_r,
        "goals": [20, 35],
        "car_params": car_params,
        "theta_indices": [2, 7]
    },
    "car2": {
        "position_indices": [(0,1), (5, 6)],
        "player_id": 1, 
        "road_logic": [1, 0, 0, 0, 0],
        "road_rules": road_rules,
        "collision_r": collision_r,
        "goals": [20, 0],
        "car_params": car_params,
        "theta_indices": [2, 7]
    }
}
###################
# Create environment:
car1_goal_cost = ProximityToBlockCost(g_params["car1"])
car2_goal_cost = ProximityToBlockCost(g_params["car2"])

# Player ids
car1_player_id = 0
car2_player_id = 1

# Build up total costs for both players. This is basically a zero-sum game.
car1_cost = PlayerCost()
car1_cost.add_cost(car1_goal_cost, "x", 1.0)

car2_cost = PlayerCost()
car2_cost.add_cost(car2_goal_cost, "x", 1.0)

_renderable_costs = [car1_goal_cost]
_player_linestyles = [".-white", ".-r", ".-b"]

if not os.path.exists("animation_tmp"):
    os.makedirs("animation_tmp")

for i in range(len(data)):
    state = data.iloc[i].to_numpy()
    plt.figure(0, figsize=(8, 10))
    _plot_lims = [-5, 25, 0,  40]

    ax = plt.gca()
    ax.set_xlabel("$x(t)$")
    ax.set_ylabel("$y(t)$")

    if _plot_lims is not None:
        ax.set_xlim(_plot_lims[0], _plot_lims[1])
        ax.set_ylim(_plot_lims[2], _plot_lims[3])

    ax.set_aspect("equal")

    # Render all costs.
    for cost in _renderable_costs:
        cost.render(ax)

    x_max = 25
    y_max = 40

    plt.title("ILQ solver solution")

    # plt.imshow(
    #     plt.imread("visual_components/grass-background-2.png", format="png"), 
    #     interpolation='none',
    #     origin='lower',
    #     extent=_plot_lims,
    #     # alpha = 0.8, 
    #     zorder = -3,
    #     clip_on=True)

    grass = plt.Rectangle(
        [-5, 0], width = 30, height = 40, color = "k", lw = 0, zorder = -2, alpha = 0.5)
    plt.gca().add_patch(grass)

    # plot road rules
    x_center = road_rules["x_min"] + 0.5 * (road_rules["x_max"] - road_rules["x_min"])
    y_center = road_rules["y_min"] + 0.5 * (road_rules["y_max"] - road_rules["y_min"])
    road = plt.Rectangle(
        [road_rules["x_min"], 0], width = road_rules["x_max"] - road_rules["x_min"], height = y_max, color = "darkgray", lw = 0, zorder = -2)
    plt.gca().add_patch(road)
    road = plt.Rectangle(
        [road_rules["x_max"], road_rules["y_min"]], width = x_max, height = road_rules["y_max"] - road_rules["y_min"], color = "darkgray", lw = 0, zorder = -2)
    plt.gca().add_patch(road)

    crosswalk_width = 3
    crosswalk_length = road_rules["x_max"] - road_rules["x_min"]
    draw_crosswalk(road_rules["x_min"], 30 - crosswalk_width*0.5, crosswalk_width, crosswalk_length)

    ax.plot([road_rules["x_min"], road_rules["x_min"]], [0, y_max], c="white", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], road_rules["x_max"]], [0, road_rules["y_min"]], c="white", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], road_rules["x_max"]], [road_rules["y_max"], y_max], c="white", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_min"], road_rules["y_min"]], c="white", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [road_rules["y_min"], road_rules["y_min"]], c="white", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_max"], road_rules["y_max"]], c="white", linewidth = 2, zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [road_rules["y_max"], road_rules["y_max"]], c="white", linewidth = 2, zorder = -1)
    ax.plot([x_center, x_center], [0, y_max], "--", c = 'white', linewidth = 5, dashes=(5, 5), zorder = -1)
    ax.plot([road_rules["x_max"], x_max], [y_center, y_center], "--", c = 'white', linewidth = 5, dashes=(5, 5), zorder = -1)

    # draw two cars using (x0, y0, theta0) and (x1, y1, theta1)
    draw_real_car(0, [state])
    if i > 15:
        draw_real_car(1, [state], path="visual_components/car_robot_y.png")
    else:
        draw_real_car(1, [state])
    # plt.plot(
    #     state[10], state[11],
    #     _player_linestyles[2],
    #     alpha = 0.4,
    #     linewidth = 2, marker='o', markersize = 10
    # )
    plt.pause(0.001)
    plt.savefig('animation_tmp/{}.jpg'.format(i)) # Trying to save these plots
    plt.clf()

# Build GIF
image_count = len(os.listdir("animation_tmp"))
with imageio.get_writer('GIF/{}.gif'.format(experiment_name), mode='I') as writer:
    for i in range(image_count):
        filename = "{}.jpg".format(i)
        image = imageio.imread(os.path.join("animation_tmp", filename))
        writer.append_data(image)