"""
Please contact the author(s) of this library if you have any questions.
Author(s): 
    Duy Phuong Nguyen (duyn@princeton.edu)
    Dennis Anthony (dennisra@princeton.edu)
"""
################################################################################
#
# Script to run a 3 player collision avoidance example intended to model
# a T-intersection.
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from cost.road_rules_penalty import RoadRulesPenalty

from resource.car_5d import Car5D
from resource.product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem
from ilq_solver.ilq_solver_threeplayer import ILQSolver
from cost.proximity_to_block_cost import ProximityToLeftBlockCost, ProximityToUpBlockCost
from cost.pedestrian_proximity_to_block_cost import PedestrianProximityToBlockCost
from player_cost.player_cost import PlayerCost
from resource.unicycle_4d import Unicycle4D

from utils.visualizer import Visualizer
from utils.logger import Logger
import math

import time
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")
datestr = time.strftime("%Y-%m-%d")

def three_player(args):
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

    # Create dynamics.
    car1 = Car5D(2.413)
    car2 = Car5D(2.413)
    ped = Unicycle4D()

    dynamics = ProductMultiPlayerDynamicalSystem(
        [car1, car2, ped], T=TIME_RESOLUTION)
    
    car1_x0 = (np.array(args.init_states)[:5]).reshape(5, 1)
    car2_x0 = (np.array(args.init_states)[5:10]).reshape(5, 1)
    ped_x0 = (np.array(args.init_states)[10:]).reshape(4, 1)

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
            "goals": args.block_goal[:2],
            "car_params": car_params,
            "theta_indices": [2, 7, 12],
            "phi_index": 3,
            "vel_index": 4
        },
        "car2": {
            "position_indices": [(0,1), (5, 6), (10, 11)],
            "player_id": 1, 
            "road_logic": [1, 0, 0, 1, 0],
            "road_rules": road_rules,
            "collision_r": collision_r,
            "goals": args.block_goal[2:4],
            "car_params": car_params,
            "theta_indices": [2, 7, 12],
            "phi_index": 8,
            "vel_index": 9
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

    config = {
        "args": args,
        "g_params": g_params,
        "l_params": None,
        "experiment": {
            "name": EXP_NAME,
            "result_dir": RESULT_DIRECTORY,
            "log_dir": LOG_DIRECTORY,
            "figure_dir": FIGURE_DIRECTORY
        }
    }
    ###################

    stacked_x0 = np.concatenate([car1_x0, car2_x0, ped_x0], axis=0)

    car1_Ps = [np.zeros((car1._u_dim, dynamics._x_dim))] * HORIZON_STEPS
    car2_Ps = [np.zeros((car2._u_dim, dynamics._x_dim))] * HORIZON_STEPS
    ped_Ps = [np.zeros((ped._u_dim, dynamics._x_dim))] * HORIZON_STEPS

    car1_alphas = [np.zeros((car1._u_dim, 1))] * HORIZON_STEPS
    car2_alphas = [np.zeros((car2._u_dim, 1))] * HORIZON_STEPS
    ped_alphas = [np.zeros((ped._u_dim, 1))] * HORIZON_STEPS

    # Create environment:
    car1_position_indices_in_product_state = (0, 1)
    car1_goal_cost = ProximityToUpBlockCost(g_params["car1"])

    # Environment for Car 2
    car2_position_indices_in_product_state = (5, 6)
    car2_goal_cost = ProximityToLeftBlockCost(g_params["car2"])

    # Environment for Pedestrian
    ped_position_indices_in_product_state = (10, 11)
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

    visualizer = Visualizer(
        [car1_position_indices_in_product_state, car2_position_indices_in_product_state, ped_position_indices_in_product_state],
        [car1_goal_cost, car2_goal_cost, ped_goal_cost
        # RoadRulesPenalty(g_params["car2"])
        ],
        [".-g", ".-r", ".-b"],
        1,
        False,
        plot_lims=[-5, 25, 0,  40],
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
                    [car1_cost, car2_cost, ped_cost],
                    stacked_x0,
                    [car1_Ps, car2_Ps, ped_Ps],
                    [car1_alphas, car2_alphas, ped_alphas],
                    0.1,
                    None,
                    logger,
                    visualizer,
                    None,
                    config)

    solver.run()