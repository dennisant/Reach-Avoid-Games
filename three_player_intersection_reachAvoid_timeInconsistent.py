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
# Script to run a 3 player collision avoidance example intended to model
# a T-intersection.
#
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt

from car_5d import Car5D
from product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem
from ilq_solver_threeplayer_cooperative_time_inconsistent_refactored import ILQSolver
from proximity_cost_reach_avoid_twoplayer import PedestrianProximityToBlockCost, ProximityToBlockCost
from player_cost_reachavoid_timeinconsistent import PlayerCost
from unicycle_4d import Unicycle4D

from visualizer import Visualizer
from logger import Logger
import math

# General parameters.
TIME_HORIZON = 3.0    # s #Change back to 2.0
TIME_RESOLUTION = 0.1 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = "./logs/three_player_time_inconsistent/"

car1 = Car5D(2.413)
car2 = Car5D(2.413)
ped = Unicycle4D()

dynamics = ProductMultiPlayerDynamicalSystem(
    [car1, car2, ped], T=TIME_RESOLUTION)

car1_theta0 = np.pi / 2.01 # 90 degree heading
car1_v0 = 5.0             # 5 m/s initial speed
car1_x0 = np.array([
    [7.25],
    [0.0],
    [car1_theta0],
    [0.0],
    [car1_v0]
])

car2_theta0 = -np.pi / 2.01 # -90 degree heading
car2_v0 = 10.0              # 2 m/s initial speed
car2_x0 = np.array([
    [3.75],
    [40.0],
    [car2_theta0],
    [0.0],
    [car2_v0]
])

ped_theta0 = 0.0 # moving right at 0.25 m/s
ped_v0 = 2.0   # moving normal to traffic flow
ped_x0 = np.array([
   [-2.0],
   [30.0],
   [ped_theta0],
   [ped_v0]
])

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

stacked_x0 = np.concatenate([car1_x0, car2_x0, ped_x0], axis=0)

car1_Ps = [np.zeros((car1._u_dim, dynamics._x_dim))] * HORIZON_STEPS
car2_Ps = [np.zeros((car2._u_dim, dynamics._x_dim))] * HORIZON_STEPS
ped_Ps = [np.zeros((ped._u_dim, dynamics._x_dim))] * HORIZON_STEPS

car1_alphas = [np.zeros((car1._u_dim, 1))] * HORIZON_STEPS
car2_alphas = [np.zeros((car2._u_dim, 1))] * HORIZON_STEPS
ped_alphas = [np.zeros((ped._u_dim, 1))] * HORIZON_STEPS

# Create environment:
car1_position_indices_in_product_state = (0, 1)
car1_goal_cost = ProximityToBlockCost(g_params["car1"])

# Environment for Car 2
car2_position_indices_in_product_state = (5, 6)
car2_goal_cost = ProximityToBlockCost(g_params["car2"])

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
    # RoadRulesPenalty(g_params["car1"])
    ],
    [".-g", ".-r", ".-b"],
    1,
    False,
    plot_lims=[-5, 25, 0,  40],
    draw_roads = True, 
    draw_cars = True,
    draw_human = True
)

# Logger.
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logger = Logger(os.path.join(LOG_DIRECTORY, 'intersection_car_example.pkl'))

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
                   None)

solver.run()