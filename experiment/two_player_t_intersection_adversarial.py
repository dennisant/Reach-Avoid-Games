"""
BSD 3-Clause License

Copyright (c) 2019, HJ Reachability Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): David Fridovich-Keil ( dfk@eecs.berkeley.edu )
"""
################################################################################
#
# Script to run a 2 player collision avoidance example intended to model
# a T-intersection.
#
################################################################################

import os
import numpy as np

from resource.car_5d import Car5D
from resource.car_10d import Car10D
from resource.product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem
from ilq_solver.ilq_solver_twoplayer_adversarial import ILQSolver
from cost.proximity_to_block_cost import ProximityToDownBlockCost, ProximityToUpBlockCost
from player_cost.player_cost import PlayerCost

from utils.visualizer import Visualizer
from utils.logger import Logger
import math

import time
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")
datestr = time.strftime("%Y-%m-%d")

def two_player_adversarial(args):
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

    # create dynamics
    car1 = Car5D(2.413)
    car2 = Car5D(2.413)

    dynamics = ProductMultiPlayerDynamicalSystem(
        [car1, car2], T=TIME_RESOLUTION)
        
    car3 = Car10D(2.413)
    dynamics_adversarial = ProductMultiPlayerDynamicalSystem(
        [car3], T=TIME_RESOLUTION)

    car1_x0 = (np.array(args.init_states)[:5]).reshape(5, 1)
    car2_x0 = (np.array(args.init_states)[5:10]).reshape(5, 1)

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
            "road_logic": [0, 1, 0, 1, 0],
            "road_rules": road_rules,
            "collision_r": collision_r,
            "goals": [20, 35],
            "car_params": car_params,
            "theta_indices": [2, 7]
        },
        "car2": {
            "position_indices": [(0,1), (5, 6)],
            "player_id": 1, 
            "road_logic": [1, 0, 0, 1, 0],
            "road_rules": road_rules,
            "collision_r": collision_r,
            "goals": [20, 0],
            "car_params": car_params,
            "theta_indices": [2, 7]
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

    stacked_x0 = np.concatenate([car1_x0, car2_x0], axis=0)

    car1_Ps = [np.zeros((car1._u_dim, dynamics._x_dim))] * HORIZON_STEPS
    car2_Ps = [np.zeros((car2._u_dim, dynamics._x_dim))] * HORIZON_STEPS

    car1_alphas = [np.zeros((car1._u_dim, 1))] * HORIZON_STEPS
    car2_alphas = [np.zeros((car2._u_dim, 1))] * HORIZON_STEPS

    # Create environment:
    car1_position_indices_in_product_state = (0, 1)
    car1_goal_cost = ProximityToUpBlockCost(g_params["car1"])

    # Environment for Car 2
    car2_position_indices_in_product_state = (5, 6)
    car2_goal_cost = ProximityToDownBlockCost(g_params["car2"])

    # Player ids
    car1_player_id = 0
    car2_player_id = 1

    # Build up total costs for both players. This is basically a zero-sum game.
    car1_cost = PlayerCost()
    car1_cost.add_cost(car1_goal_cost, "x", 1.0)

    car2_cost = PlayerCost()
    car2_cost.add_cost(car2_goal_cost, "x", 1.0)

    visualizer = Visualizer(
        [car1_position_indices_in_product_state, car2_position_indices_in_product_state],
        [car1_goal_cost, car2_goal_cost],
        [".-g", ".-r", ".-b"],
        1,
        False,
        plot_lims=[-5, 25, -2,  40],
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

    # Set up ILQSolver.
    solver = ILQSolver(
        dynamics,
        [car1_cost, car2_cost],
        stacked_x0,
        [car1_Ps, car2_Ps],
        [car1_alphas, car2_alphas],
        0.1,
        None,
        logger,
        visualizer,
        None,
        config,
        dynamics_adversarial
    )

    solver.run()