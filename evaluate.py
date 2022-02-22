# create the rollout animation of the three-player game using the log trajectory
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation, markers
import os
from matplotlib.transforms import Affine2D
from cost.proximity_to_block_cost import ProximityToLeftBlockCost, ProximityToUpBlockCost
from resource.car_5d import Car5D
from cost.pedestrian_proximity_to_block_cost import PedestrianProximityToBlockCost
from player_cost.player_cost import PlayerCost
from resource.unicycle_4d import Unicycle4D

import math
import pandas as pd
import imageio

from utils.utils import draw_crosswalk, draw_real_car, draw_real_human

parser = argparse.ArgumentParser()
parser.add_argument("--evaluate",       help="Things to evaluate",       choices=["train", "rollout"],        required=True)
parser.add_argument("--loadpath",       help="Path of experiment",       required=True)
parser.add_argument("--iteration",      help="Iteration of experiment to evaluate",     type=int)
args = parser.parse_args()

loadpath = args.loadpath

if not os.path.exists(loadpath):
    raise ValueError("Experiment does not exist")

def train_process():
    folder_path = os.path.join(loadpath, "figures")

    if not os.path.exists(folder_path):
        raise ValueError("There is no such path: {}, please check again".format(folder_path))

    # Build GIF
    image_count = len([f for f in os.listdir(folder_path) if "plot-" in f])
    with imageio.get_writer('{}/evaluate_training.gif'.format(folder_path), mode='I') as writer:
        for i in range(image_count):
            filename = "plot-{}.jpg".format(i)
            image = imageio.imread(os.path.join(folder_path, filename))
            writer.append_data(image)

def final_rollout():
    # check to see if there is logs folder:
    if not ("logs" in os.listdir(loadpath)):
        raise ValueError("There is no log folder in this experiment")

    # get experiment file:
    file_list = os.listdir(os.path.join(loadpath, "logs"))
    print("\t>> Found {} file(s)".format(len(file_list)))

    if len(file_list) > 1:
        index = input("Please choose which log file to use: ")
    else: 
        index = 0

    # Read log
    file_path = os.path.join(loadpath, "logs", file_list[index])
    with open(file_path, "rb") as log:
        raw_data = pickle.load(log)

    if args.iteration is None:
        print("\t>> Get the last iteration to render")
        iteration = np.array(raw_data["xs"]).shape[0] - 1
    else:
        iteration = args.iteration

    print("\t>> Iteration to render: {}".format(iteration))

    # create output folder
    output = os.path.join(loadpath, "evaluate")
    if not os.path.exists(output):
        os.makedirs(output)
    print("\t>> Output folder: " + output)

    data = pd.DataFrame(
        np.array(raw_data["xs"][iteration]).reshape((
            len(raw_data["xs"][iteration]), 14
        )), columns = [
            "x0", "y0", "theta0", "phi0", "vel0",
            "x1", "y1", "theta1", "phi1", "vel1",
            "x2", "y2", "theta2", "vel2"
        ]
    )

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
        },
        "ped1": {
            "position_indices": [(0,1), (5, 6), (10, 11)],
            "player_id": 2, 
            "road_logic": [1, 1, 1, 1, 0],
            "road_rules": ped_road_rules,
            "goals": [15, 30],
            "car_params": car_params,
            "theta_indices": [2, 7, 12]
        }
    }
    ###################
    # Create environment:
    car1_goal_cost = ProximityToUpBlockCost(g_params["car1"])
    car2_goal_cost = ProximityToLeftBlockCost(g_params["car2"])
    ped_goal_cost = PedestrianProximityToBlockCost(g_params["ped1"])

    # Player ids
    car1_player_id = 0
    car2_player_id = 1
    ped1_player_id = 2

    # Build up total costs for both players. This is basically a zero-sum game.
    car1_cost = PlayerCost()
    car1_cost.add_cost(car1_goal_cost, "x", 1.0)

    car2_cost = PlayerCost()
    car2_cost.add_cost(car2_goal_cost, "x", 1.0)

    ped_cost = PlayerCost()
    ped_cost.add_cost(ped_goal_cost, "x", 1.0)
        
    _renderable_costs = [car1_goal_cost, car2_goal_cost, ped_goal_cost]

    for i in range(len(data)):
        state = data.iloc[i].to_numpy()
        plt.figure(0)
        _plot_lims = [-5, 25, 0, 40]

        ratio = (_plot_lims[1] - _plot_lims[0])/(_plot_lims[3] - _plot_lims[2])
        plt.gcf().set_size_inches(ratio*8, 8)

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

        draw_real_car(0, [state])
        draw_real_car(1, [state])
        draw_real_human([state], i%2)
        plt.pause(0.001)
        plt.savefig(os.path.join(output, 'step-{}.jpg'.format(i))) # Trying to save these plots
        plt.clf()

    # Build GIF
    image_count = len([f for f in os.listdir(output) if "step-" in f])
    with imageio.get_writer(os.path.join(output, 'evaluate_rollout.gif'), mode='I') as writer:
        try:
            for i in range(image_count):
                filename = "step-{}.jpg".format(i)
                image = imageio.imread(os.path.join(output, filename))
                writer.append_data(image)
        except FileNotFoundError:
            pass

if args.evaluate == "train":
    print("\t>> Evaluate the training process")
    train_process()
else:
    print("\t>> Evaluate the final rollout")
    final_rollout()