import os
import numpy as np
import matplotlib.pyplot as plt

from resource.car_5d import Car5D
from cost.obstacle_penalty import ObstacleDistCost
from resource.product_multiplayer_dynamical_system import \
    ProductMultiPlayerDynamicalSystem
from ilq_solver.ilq_solver_oneplayer_cooperative_time_consistent_refactored import ILQSolver
from cost.proximity_cost_reach_avoid_twoplayer import ProximityCost
from player_cost.player_cost import PlayerCost

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

EXP_NAME = "one_player_time_consistent"
LOG_DIRECTORY = "./result/" + EXP_NAME + "_" + timestr + "/"

car = Car5D(2.413)

dynamics = ProductMultiPlayerDynamicalSystem(
    [car], T=TIME_RESOLUTION)

car_theta0 = np.pi / 2.01
car_v0 = 8.0
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
            # (6.0, 25.0)
        ],
        "obstacle_radii": [
            4.5, 3.0, 3.0
            # 4.0
        ]
    }
}

l_params = {
    "car": {
        "goals": [
            (6.0, 40.0)
        ],
        "goal_radii": [
            2.0
        ]
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