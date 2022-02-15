"""
x: [-20, 30]
y: [0, 60]
theta: [0, 2*pi]
v: [3, 10]
Time horizon: {3.0, 4.0, 5.0, ..., 10.0}
Constant Goal:      (x, y, radius)
                6.0, 40.0, 2.0
Constant Obstacles: (x, y, radius)
                9.0, 25.0, 4.5
	            20.0, 35.0, 3.0
	            6.5, 50.0, 3.0
                1.0, 38.0, 3.0
	           -1.0, 42.0, 2.0
                2.5, 50.0, 2.0
Max runtime before termination: 150
"""

import numpy as np
import os

# Constant params
base_flag = "   python3 run.py                      \
                --time_consistency                  \
                --no_players 1                      \
                --env_type goal_with_obs            \
                --eps_control 0.1 --eps_state 0.1   \
                --linesearch                        \
                --linesearch_type trust_region      \
            "

obstacle_flag = "--obstacles            \
                    9.0, 25.0, 4.5      \
                    20.0, 35.0, 3.0     \
	                6.5, 50.0, 3.0      \
                    1.0, 38.0, 3.0      \
	                -1.0, 42.0, 2.0     \
                    2.5, 50.0, 2.0      \
                "

goal_flag = "--goal 6.0 40.0 2.0"

# Dynamic params
no_of_runs = 100
x = np.random.uniform(-20, 30, no_of_runs)
y = np.random.uniform(0, 60, no_of_runs)
theta = np.random.uniform(0, 2*np.pi, no_of_runs)
v = np.random.uniform(3, 10, no_of_runs)
time_horizon = np.random.randint(3, 11, no_of_runs)
max_runtime = 150

for i in range(no_of_runs):
    init_state_flag = "--init_states {} {} {} 0.0 {}".format(x[i], y[i], theta[i], v[i])
    time_horizon_flag = "--time_horizon {}".format(time_horizon[i])
    max_runtime_flag = "--max_steps {}".format(max_runtime)

    cmd = base_flag + obstacle_flag + goal_flag + init_state_flag + time_horizon_flag + max_runtime_flag
    os.system(cmd)

