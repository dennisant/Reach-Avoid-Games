import numpy as np
import os
import pickle
import pandas as pd
from cost.obstacle_penalty import ObstacleDistCost
from cost.proximity_cost import ProximityCost
import matplotlib.pyplot as plt
from utils.visualizer import Visualizer

batch_name = "batch-2022-02-15"
is_converged = []
xs = []

if not os.path.exists("result/" + batch_name):
    raise ValueError("Batch does not exist: " + batch_name)

print("Collecting all experiments in batch")
list_of_experiments = sorted([dir for dir in os.listdir("result/" + batch_name) if "experiment" in dir])
print("\t>> Found {} experiments".format(len(list_of_experiments)))
print("Collecting status and final trajectory of each experiment")
for exp in list_of_experiments:
    log_path = os.path.join("result", batch_name, exp, "logs", "experiment.pkl")
    
    with open(log_path, "rb") as log:
        data = pickle.load(log)
    
    # get is_converged data
    if "is_converged" in data.keys():
        is_converged.append(True)
    else:
        is_converged.append(False)

    # get state data
    if "xs" in data.keys():
        xs.append(data["xs"])
    else:
        raise ValueError("Cannot find state data")
    
visualizer = Visualizer(
    [(0, 1)],
    [ProximityCost(data["l_params"][0]["car"], data["g_params"][0]["car"]), ObstacleDistCost(data["g_params"][0]["car"])],
    ["-b", ".-r", ".-g"],
    1,
    False,
    plot_lims=[-20, 75, -20,  100],
    draw_cars = False
)

for i in range(len(xs)):
    visualizer.add_trajectory(None, {"xs": xs[i][-1]})
    visualizer.draw_real_car(0, np.array(xs[i][-1])[[0]])
    visualizer.plot(alpha = 0.4)
plt.show()
        