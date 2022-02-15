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
import logging
import time
datestr = time.strftime("%Y-%m-%d")

if os.path.exists("result/batch-{}/".format(datestr)):
    raise ValueError("You are running a new batch onto an existing one")
else:
    os.makedirs("result/batch-{}/".format(datestr))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-20s '
            '%(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler(
            "result/batch-{}/batch_test.log".format(datestr)),
        logging.StreamHandler()])

# Constant params
base_flag = "   python3 run.py                      \
                --no_players 1                      \
                --env_type goal_with_obs            \
                --eps_control 0.1 --eps_state 0.1   \
                --linesearch                        \
                --linesearch_type trust_region      \
                --batch_run                         \
                --plot                              \
                --log                               \
            "

obstacle_flag = " --obstacles         \
                    9.0 25.0 4.5      \
                    20.0 35.0 3.0     \
	                6.5 50.0 3.0      \
                    1.0 38.0 3.0      \
	                -1.0 42.0 2.0     \
                    2.5 50.0 2.0      \
                "

goal_flag = " --goal 6.0 40.0 2.0"

# Dynamic params
no_of_runs = 5
x = np.random.uniform(-20, 30, no_of_runs)
y = np.random.uniform(0, 60, no_of_runs)
theta = np.random.uniform(0, 2*np.pi, no_of_runs)
v = np.random.uniform(3, 10, no_of_runs)
time_horizon = np.random.randint(3, 11, no_of_runs)
max_runtime = 150
time_consistency = False

for i in range(no_of_runs):
    init_state_flag = " --init_states {} {} {} 0.0 {}".format(x[i], y[i], theta[i], v[i])
    time_horizon_flag = " --t_horizon {}".format(time_horizon[i])
    max_runtime_flag = " --max_steps {}".format(max_runtime)

    cmd = base_flag + obstacle_flag + goal_flag + init_state_flag + time_horizon_flag + max_runtime_flag
    if time_consistency:
        cmd = cmd + " --time_consistency"
    cmd = " ".join(cmd.split())
    logging.info("Running test#{} with data: \n\r\t\n".format(i) + cmd.strip("\t"))
    os.system(cmd)

