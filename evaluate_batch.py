import numpy as np
import os
import pickle
import pandas as pd
from cost.obstacle_penalty import ObstacleDistCost
from cost.proximity_cost import ProximityCost
import matplotlib.pyplot as plt
from utils.visualizer import Visualizer

batch_name = "batch-2022-02-17"
exp_suffix = "exp_time_consistent"
is_converged = []
xs = []

if not os.path.exists("result/" + batch_name):
    raise ValueError("Batch does not exist: " + batch_name)

print("Collecting all experiments in batch")
list_of_experiments = sorted([dir for dir in os.listdir("result/" + batch_name) if exp_suffix in dir])
print("\t>> Found {} experiments".format(len(list_of_experiments)))
print("Collecting status and final trajectory of each experiment")
for exp in list_of_experiments:
    print("\t\t>> Loading experiment: {}".format(exp))
    log_path = os.path.join("result", batch_name, exp, "logs", "{}.pkl".format(exp_suffix))
    
    try:
        with open(log_path, "rb") as log:
            data = pickle.load(log)
    except EOFError:
        print("\t\t\t>> Dropping exp {} due to EOFError, please manually check the file".format(exp))
        continue
    
    # get is_converged data
    if "is_converged" in data.keys():
        if data["is_converged"][0]:
            is_converged.append(True)
            # get state data
            if "xs" in data.keys():
                xs.append(data["xs"])
            else:
                raise ValueError("Cannot find state data")
        else:
            is_converged.append(False)
    else:
        is_converged.append(False)

convergence_rate = len([i for i in is_converged if i is True])/len(is_converged)
print("\t>> Convergence rate: {:.3f} ({} of {} runs)".format(convergence_rate, len([i for i in is_converged if i is True]), len(is_converged))) 

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
    plt.scatter(np.array(xs[i][-1])[[0]][0][0], np.array(xs[i][-1])[[0]][0][1], color="firebrick", zorder=10)
    plt.scatter(np.array(xs[i][-1])[[-1]][0][0], np.array(xs[i][-1])[[-1]][0][1], color="aqua", zorder=10, alpha = 0.4)
    # visualizer.draw_real_car(0, np.array(xs[i][-1])[[0]])
    visualizer.plot(alpha = 0.4, base_size = 20.0)

summary_dir = os.path.join(os.getcwd(), "result", batch_name, "summary")
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

plt.savefig(summary_dir + "/" + exp_suffix + "_summary.svg", format="svg")
plt.savefig(summary_dir + "/" + exp_suffix + "_summary.png", format="png")
plt.show()
