"""
Please contact the author(s) of this library if you have any questions.
Author(s): 
    Duy Phuong Nguyen (duyn@princeton.edu)
    Dennis Anthony (dennisra@princeton.edu)
"""
################################################################################
#
# Script to run a 1 player collision avoidance example
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from resource.car_5d import Car5D
from cost.obstacle_penalty import ObstacleDistCost
from resource.product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem
from ilq_solver.ilq_solver_oneplayer import ILQSolver
from cost.proximity_cost import ProximityCost
from player_cost.player_cost import PlayerCost

from utils.visualizer import Visualizer
from utils.logger import Logger
import math
from resource.point import Point
from utils.argument import get_argument
import time
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")
datestr = time.strftime("%Y-%m-%d")

def one_player(args):
    # General parameters.
    TIME_HORIZON = args.t_horizon
    TIME_RESOLUTION = args.t_resolution
    HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)

    EXP_NAME = args.exp_name
    if args.batch_run:
        RESULT_DIRECTORY = "./result/batch-" + datestr + "/" + EXP_NAME + "_" + timestr + "/"
    else:
        RESULT_DIRECTORY = "./result/" + EXP_NAME + "_" + timestr + "/"
    LOG_DIRECTORY = RESULT_DIRECTORY + "logs/"
    FIGURE_DIRECTORY = RESULT_DIRECTORY + "figures/"

    car = Car5D(2.413)

    dynamics = ProductMultiPlayerDynamicalSystem(
        [car], T=TIME_RESOLUTION)

    car_x0 = (np.array(args.init_states)[:5]).reshape(5, 1)

    ###################
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }

    collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

    g_params = {
        "car": {
            "position_indices": [(0,1)],
            "player_id": 0, 
            "collision_r": collision_r,
            "car_params": car_params,
            "theta_indices": [2],
            "phi_index": 3, 
            "vel_index": 4,
            "obstacles": list(zip(*(iter([np.array(args.obstacles)[i] for i in range(len(args.obstacles)) if i % 3 < 2]),) * 2)),
            "obstacle_radii": [np.array(args.obstacles)[i] for i in range(len(args.obstacles)) if i % 3 == 2]
        }
    }

    l_params = {
        "car": {
            "goals": list(zip(*(iter(args.goal[0:2]),) * 2)),
            "goal_radii": [args.goal[2]]
        }
    }

    config = {
        "g_params": g_params,
        "l_params": l_params,
        "experiment": {
            "name": EXP_NAME,
            "result_dir": RESULT_DIRECTORY,
            "log_dir": LOG_DIRECTORY,
            "figure_dir": FIGURE_DIRECTORY
        },
        "args": args
    }
    ###################

    stacked_x0 = np.concatenate([car_x0], axis=0)

    car_Ps = [np.zeros((car._u_dim, dynamics._x_dim))] * HORIZON_STEPS
    car_alphas = [np.zeros((car._u_dim, 1))] * HORIZON_STEPS

    # Create environment:
    car_position_indices_in_product_state = (0, 1)
    for i in range(len(l_params["car"]["goals"])):
        car_goal_cost = ProximityCost(l_params["car"], g_params["car"])

    # Player ids
    car_player_id = 0

    car_cost = PlayerCost(**vars(args))
    car_cost.add_cost(car_goal_cost, "x", 1.0)

    obstacle_costs = [ObstacleDistCost(g_params["car"])]

    visualizer = Visualizer(
        [car_position_indices_in_product_state],
        [car_goal_cost] + obstacle_costs,
        ["-b", ".-r", ".-g"],
        1,
        False,
        plot_lims=[-20, 75, -20,  100],
        **vars(args)
    )

    # Logger.
    if args.log or args.plot:
        if not os.path.exists(RESULT_DIRECTORY):
            os.makedirs(RESULT_DIRECTORY)
        if args.log and not os.path.exists(LOG_DIRECTORY):
            os.makedirs(LOG_DIRECTORY)
        if args.plot and not os.path.exists(FIGURE_DIRECTORY):
            os.makedirs(FIGURE_DIRECTORY)

    if args.log:
        logger = Logger(os.path.join(LOG_DIRECTORY, EXP_NAME + '.pkl'))
    else:
        logger = None

    if args.note != "":
        if not os.path.exists(RESULT_DIRECTORY):
            os.makedirs(RESULT_DIRECTORY)
        with open(os.path.join(RESULT_DIRECTORY, "note.txt"), 'w') as file:
            file.write(args.note)

    # Set up ILQSolver.
    solver = ILQSolver(dynamics,
                    [car_cost],
                    stacked_x0,
                    [car_Ps],
                    [car_alphas],
                    0.1,
                    None,
                    logger,
                    visualizer,
                    None, 
                    config)

    solver.run()