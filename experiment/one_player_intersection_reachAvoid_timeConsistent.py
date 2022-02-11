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
timestr = time.strftime("%Y-%m-%d-%H_%M")

def one_player_time_consistent(args):
    # General parameters.
    TIME_HORIZON = args.t_horizon
    TIME_RESOLUTION = args.t_resolution
    HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)

    EXP_NAME = args.exp_name
    LOG_DIRECTORY = "./result/" + EXP_NAME + "_" + timestr + "/"

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
            "log_dir": LOG_DIRECTORY
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
        car_goal_cost = ProximityCost(
            car_position_indices_in_product_state,
            l_params["car"]["goals"][i],
            l_params["car"]["goal_radii"][i],
            name="car_goal"
        )

    # Player ids
    car_player_id = 0

    car_cost = PlayerCost()
    car_cost.add_cost(car_goal_cost, "x", 1.0)

    obstacle_costs = [ObstacleDistCost(g_params["car"])]

    visualizer = Visualizer(
        [car_position_indices_in_product_state],
        [car_goal_cost] + obstacle_costs,
        ["-b", ".-r", ".-g"],
        1,
        False,
        plot_lims=[-20, 75, -20,  100],
        draw_cars = args.draw_cars
    )

    # Logger.
    if args.log or args.plot:
        if not os.path.exists(LOG_DIRECTORY):
            os.makedirs(LOG_DIRECTORY)

    if args.log:
        logger = Logger(os.path.join(LOG_DIRECTORY, EXP_NAME + '.pkl'))
    else:
        logger = None

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