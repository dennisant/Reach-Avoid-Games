"""
x: [-20, 30]
y: [-2, 60]
theta: [0, 2*pi]
v: [3, 10]
Time horizon: {3.0, 4.0, 5.0, ..., 10.0}
Constant Goal:      (x, y, radius)
                6.0, 40.0, 2.0
Constant Obstacles: (x, y, radius)
                9.0, 25.0, 4.5
                20.0, 35.0, 3.0
	            6.5, 50.0, 3.0
	            -4.0, 33.0, 2.0
	            -5.0, 44.0, 2.0
                2.5, 50.0, 2.0
Max runtime before termination: 150
"""
from bdb import effective
import numpy as np
import os
import logging
import time
import math
datestr = time.strftime("%Y-%m-%d")

if os.path.exists("result/batch-{}/".format(datestr)):
    raise ValueError("You are running a new batch onto an existing one, please append name or delete old run before running")
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

car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }

collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

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
                --hallucinated                      \
            "

obstacle_flag = " --obstacles             \
                    9.0 25.0 4.5          \
                    20.0 35.0 3.0         \
                    6.5 50.0 3.0          \
                    -4.0 33.0 2.0         \
                    -5.0 44.0 2.0         \
                    2.5 50.0 2.0          \
                "

goal_flag = " --goal 6.0 40.0 2.0"

# Dynamic params
no_of_runs = 10
v = np.random.uniform(3, 10, no_of_runs)
time_horizon = np.random.randint(3, 11, no_of_runs)
max_runtime = 150
# time_consistent, time_inconsistent, both
run_type = "both"

obstacles = [float(i) for i in obstacle_flag.replace("--obstacles", "").split()]
goal = [float(i) for i in goal_flag.replace("--goal", "").split()]
goal_and_obs = obstacles + goal

def get_random_initial_pos():
    x = np.random.uniform(-20, 30)
    y = np.random.uniform(-2, 60)
    return x, y

def format_angle(angle):
    """
    Format angle to be within the range of [-pi,pi]
    """
    while angle > np.pi or angle < -np.pi:
        if angle > np.pi:
            angle = angle - 2*np.pi
        elif angle < -np.pi:
            angle = angle + 2*np.pi
    return angle

def run_time_consistent(cmd):
    cmd = cmd + " --time_consistency --exp_name exp_time_consistent"
    cmd = " ".join(cmd.split())
    logging.info("Running test#{} with data: \n\r\t\n".format(i) + cmd.strip("\t"))
    os.system(cmd)

def run_time_inconsistent(cmd):
    cmd = cmd + " --exp_name exp_time_inconsistent"
    cmd = " ".join(cmd.split())
    logging.info("Running test#{} with data: \n\r\t\n".format(i) + cmd.strip("\t"))
    os.system(cmd)

for i in range(no_of_runs):
    found_good_initial_pos = False

    # check if the sampled x, y are too close to obstacle or the goal
    while not found_good_initial_pos:
        x, y = get_random_initial_pos()
        found_good_initial_pos = True
        for j in range(int(len(goal_and_obs)/3.0)):
            dx = x - goal_and_obs[j*3]
            dy = y - goal_and_obs[j*3 + 1]
            r = goal_and_obs[j*3 + 2]
            relative_distance = math.sqrt(dx*dx + dy*dy)
            if r - relative_distance + 3.0 * collision_r > 0:
                # the initial position collides with an obstacle
                found_good_initial_pos = False
                break

    # calculate the effective theta heading of the car to the goal
    effective_theta = math.atan2(40.0 - y, 6.0 - x)
    
    # sampling theta in the range [-pi/4 + effective_theta, pi/4 + effective theta]
    theta = format_angle(np.random.uniform(-np.pi/4.0, np.pi/4.0) + effective_theta)
    while abs(theta - format_angle(effective_theta)) < np.pi/60.0:
        theta = format_angle(np.random.uniform(-np.pi/4.0, np.pi/4.0) + effective_theta)

    init_state_flag = " --init_states {} {} {} 0.0 {}".format(x, y, theta, v[i])
    time_horizon_flag = " --t_horizon {}".format(time_horizon[i])
    max_runtime_flag = " --max_steps {}".format(max_runtime)

    cmd = base_flag + obstacle_flag + goal_flag + init_state_flag + time_horizon_flag + max_runtime_flag
    if run_type == "time_consistent":
        run_time_consistent(cmd)
    elif run_type == "time_inconsistent":
        run_time_inconsistent(cmd)
    elif run_type == "both":
        run_time_consistent(cmd)
        run_time_inconsistent(cmd)
    else:
        raise NotImplementedError

