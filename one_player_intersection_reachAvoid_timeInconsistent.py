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
from ilq_solver.ilq_solver_oneplayer_cooperative_time_inconsistent_refactored import ILQSolver
from cost.proximity_cost_reach_avoid_twoplayer import ProximityCost
from player_cost.player_cost_reachavoid_timeinconsistent import PlayerCost

from utils.visualizer import Visualizer
from utils.logger import Logger
import math
from resource.point import Point

from utils.argument import get_argument

import time
timestr = time.strftime("%Y-%m-%d-%H_%M")

args = get_argument()

# General parameters.
TIME_HORIZON = 12.0    # s #Change back to 2.0
TIME_RESOLUTION = 0.1 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)

EXP_NAME = "one_player_time_inconsistent"
LOG_DIRECTORY = "./result/" + EXP_NAME + "_" + timestr + "/"

car = Car5D(2.413)

dynamics = ProductMultiPlayerDynamicalSystem(
    [car], T=TIME_RESOLUTION)

car_theta0 = np.pi / 2.01
car_v0 = 12.0
car_x0 = np.array([
    [6.0],
    [0.0],
    [car_theta0],
    [0.0],
    [car_v0]
])

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
        "obstacles": [
            (9.0, 25.0),
            (20.0, 35.0),
            (6.5, 46.0)
        ],
        "obstacle_radii": [
            4.5, 3.0, 3.0
        ]
    }
}

config = {
    "g_params": g_params,
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
car_goal_cost = ProximityCost(
    car_position_indices_in_product_state,
    (6.0, 40.0),
    2.0,
    name="car_goal"    
)

# Player ids
car_player_id = 0

# Build up total costs for both players. This is basically a zero-sum game.
car_cost = PlayerCost()
car_cost.add_cost(car_goal_cost, "x", 1.0)

# obstacle_centers = [Point(6.5, 15.0), Point(0.0, 20.0), Point(12.0, 24.0)]
obstacle_centers = [Point(6.5, 30.0)]
# obstacle_radii = [4.5, 1.5, 4.0]
obstacle_radii = [6.5]

obstacle_costs = [ObstacleDistCost(g_params["car"])]

visualizer = Visualizer(
    [car_position_indices_in_product_state],
    [car_goal_cost] + obstacle_costs,
    [".-g", ".-r", ".-b"],
    1,
    False,
    plot_lims=[-5, 35, -2,  100]
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