import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pickle
import argparse
from experiment.batch_experiment import Batch

parser = argparse.ArgumentParser()
parser.add_argument("--loadpath",       help="Path of batch",       required=True)
parser.add_argument("--exp_suffix",     help="Suffix of exp name",  required=True)

args = parser.parse_args()

loadpath = args.loadpath
exp_suffix = args.exp_suffix
batch_name = os.path.basename(os.path.abspath(loadpath))

batch = Batch(batch_name, exp_suffix)
print("\t\t>> Plot all runs")
batch.visualize_all_runs(savefig=True)

print("\t\t>> Plot success runs")
batch.visualize_all_runs(data=batch.get_success_run(), savefig=True, suffix="converged")

no_of_runs = batch.data.shape[0]
# Orange trajectories for runs that reach but then violates
# J-0 < 0, with k \in [t_star, T], exist g_k > 0
runs_successful_with_j0 = (batch.data["first_negative_cost"][np.array(np.where(batch.data["first_negative_cost"] > 0)).flatten()]).index
from cost.obstacle_penalty import ObstacleDistCost

g_func = ObstacleDistCost(batch.info["g_params"][0]["car"])
runs_successful_with_j0_with_violation = []

for i in runs_successful_with_j0:
    for j in range(batch.data.iloc[i]["first_t_star"][0], len(batch.data.iloc[i]["end_traj"])):
        if g_func(batch.data.iloc[i]["end_traj"][j])[0] > 0:
            runs_successful_with_j0_with_violation.append(i)
            break

# Red trajectories for runs that never satisfy J_0
runs_not_successful_with_j0 = sorted(list(set(range(no_of_runs)) - set(runs_successful_with_j0)))

# Blue trajectories for runs that reach, then no violations afterward
reach_then_no_violation_runs = sorted(list(set(range(no_of_runs)) - set(runs_successful_with_j0_with_violation) - set(runs_not_successful_with_j0)))

print("\t\t>> Check common set between blue and orange: {}".format(set(reach_then_no_violation_runs) & set(runs_successful_with_j0_with_violation)))
print("\t\t>> Check common set between red and orange: {}".format(set(runs_not_successful_with_j0) & set(runs_successful_with_j0_with_violation)))
print("\t\t>> Check common set between blue and red: {}".format(set(reach_then_no_violation_runs) & set(runs_not_successful_with_j0)))
print("\t\t>> Total run count: {}".format(len(runs_successful_with_j0_with_violation) + len(runs_not_successful_with_j0) + len(reach_then_no_violation_runs)))

blue_data = batch.data.iloc[reach_then_no_violation_runs]
orange_data = batch.data.iloc[runs_successful_with_j0_with_violation]
red_data = batch.data.iloc[runs_not_successful_with_j0]

# Plot three types:
batch.visualize_all_runs(data=blue_data, overlay=True, style="-b", alpha=0.6)
batch.visualize_all_runs(data=orange_data, overlay=True, style="darkorange", alpha=0.5)
batch.visualize_all_runs(data=red_data, overlay=True, style="-r", alpha=0.5)
batch.savefig(suffix="three_color_plot")