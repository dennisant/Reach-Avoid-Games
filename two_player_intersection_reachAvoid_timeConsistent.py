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

from car_5d import Car5Dv2
from product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem

from point import Point
from polyline import Polyline

from ilq_solver_twoplayer_cooperative_time_consistent_refactored import ILQSolver
from proximity_cost_reach_avoid_twoplayer import ProximityCost, ProximityToBlockCost
from semiquadratic_cost import SemiquadraticCost
from quadratic_cost import QuadraticCost
from semiquadratic_polyline_cost_draw import SemiquadraticPolylineCostDraw
from quadratic_polyline_cost import QuadraticPolylineCost
from player_cost_threeplayer_reachavoid_timeconsistent import PlayerCost

from visualizer import BlockVisualizer, Visualizer
from logger import Logger

# General parameters.
TIME_HORIZON = 3.0    # s #Change back to 2.0
TIME_RESOLUTION = 0.1 # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RESOLUTION)
LOG_DIRECTORY = "./logs/three_player/"

# Create dynamics.
car_params = {
    "wheelbase": 2.413, 
    "length": 4.267,
    "width": 1.988
}
car1 = Car5Dv2(4.0, **car_params)
car2 = Car5Dv2(4.0, **car_params)
#ped = PointMass2D()

dynamics = ProductMultiPlayerDynamicalSystem(
    [car1, car2], T=TIME_RESOLUTION)

# Choose initial states and set initial control laws to zero, such that
# we start with a situation that looks like this:
#
#              (car 2)
#             |   X   .       |
#             |   :   .       |
#             |  \./  .       |
# (unicycle) X-->     .       |
#             |       .        ------------------
#             |       .
#             |       .        ..................
#             |       .
#             |       .        ------------------
#             |       .   ^   |
#             |       .   :   |         (+y)
#             |       .   :   |          |
#             |       .   X   |          |
#                      (car 1)           |______ (+x)
#
# We shall set up the costs so that car 2 wants to turn and car 1 / unicycle 1
# continue straight in their initial direction of motion.
# We shall assume that lanes are 4 m wide and set the origin to be in the
# bottom left along the road boundary.

car1_theta0 = np.pi / 2.0 # 90 degree heading
car1_v0 = 10.0             # 5 m/s initial speed
car1_x0 = np.array([
    [6.0],
    [1],
    [car1_theta0],
    [0.0],
    [car1_v0]
])

car2_theta0 = -np.pi / 2.0 # -90 degree heading
car2_v0 = 20.0              # 2 m/s initial speed
car2_x0 = np.array([
    [3.5],
    [22.0],
    [car2_theta0],
    [0.0],
    [car2_v0]
])


#ped_vx0 = 0.25 # moving right at 0.25 m/s
#ped_vy0 = 0.0   # moving normal to traffic flow
#ped_x0 = np.array([
#    [-4.0],
#    [19.0],
#    [ped_vx0],
#    [ped_vy0]
#])


stacked_x0 = np.concatenate([car1_x0, car2_x0], axis=0)

car1_Ps = [np.zeros((car1._u_dim, dynamics._x_dim))] * HORIZON_STEPS
car2_Ps = [np.zeros((car2._u_dim, dynamics._x_dim))] * HORIZON_STEPS
#ped_Ps = [np.zeros((ped._u_dim, dynamics._x_dim))] * HORIZON_STEPS

car1_alphas = [np.zeros((car1._u_dim, 1))] * HORIZON_STEPS
car2_alphas = [np.zeros((car2._u_dim, 1))] * HORIZON_STEPS
#ped_alphas = [np.zeros((ped._u_dim, 1))] * HORIZON_STEPS


# Create environment:
car1_position_indices_in_product_state = (3.5, 22)
car1_goal_cost = ProximityToBlockCost(
    car1_position_indices_in_product_state, "car1_goal")

# Environment for Car 2
car2_position_indices_in_product_state = (6, 1)

car2_goal_cost = ProximityToBlockCost(
    car2_position_indices_in_product_state, "car2_goal")

# Penalize speed above a threshold for all players.
car1_v_index_in_product_state = 4
car1_maxv = 8.0 # m/s
car1_minv_cost = SemiquadraticCost(
    car1_v_index_in_product_state, 0.0, False, "car1_minv")
car1_maxv_cost = SemiquadraticCost(
    car1_v_index_in_product_state, car1_maxv, True, "car1_maxv")

car2_v_index_in_product_state = 9
car2_maxv = 8.0 # m/s
car2_minv_cost = SemiquadraticCost(
    car2_v_index_in_product_state, 0.0, False, "car2_minv")
car2_maxv_cost = SemiquadraticCost(
    car2_v_index_in_product_state, car2_maxv, True, "car2_maxv")


# Penalize deviation from nominal speed for all players
car1_nominalv = 0.0 # 8.0
car2_nominalv = 0.0 # 6.0

# Control costs for all players.
car1_steering_cost = QuadraticCost(0, 0.0, "car1_steering")
car1_a_cost = QuadraticCost(1, 0.0, "car1_a")

car2_steering_cost = QuadraticCost(0, 0.0, "car2_steering")
car2_a_cost = QuadraticCost(1, 0.0, "car2_a")


# Player ids
car1_player_id = 0
car2_player_id = 1

# Proximity cost.
CAR_PROXIMITY_THRESHOLD = 3.0

# Build up total costs for both players. This is basically a zero-sum game.
car1_cost = PlayerCost()
car1_cost.add_cost(car1_goal_cost, "x", 1.0) #30.0 # -1.0

car1_player_id = 0

car2_cost = PlayerCost()
car2_cost.add_cost(car2_goal_cost, "x", 1.0) #30.0 # -1.0

car2_player_id = 1

visualizer = BlockVisualizer(
    [car1_position_indices_in_product_state,
     car2_position_indices_in_product_state],
    [car1_goal_cost,
     car2_goal_cost
    ],
    [".-r", ".-g", ".-b"],
    1,
    False,
    plot_lims=[0, 20, 0, 25])

# Logger.
if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logger = Logger(os.path.join(LOG_DIRECTORY, 'intersection_car_example.pkl'))

# Set up ILQSolver.
solver = ILQSolver(dynamics,
                   [car1_cost, car2_cost],
                   stacked_x0,
                   [car1_Ps, car2_Ps],
                   [car1_alphas, car2_alphas],
                   0.1,
                   None,
                   logger,
                   visualizer,
                   [[car1_position_indices_in_product_state, car2_position_indices_in_product_state]])

solver.run()