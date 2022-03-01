import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pickle
from cost.obstacle_penalty import ObstacleDistCost
from cost.proximity_cost import ProximityCost
from utils.visualizer import Visualizer

class Batch(object):
    """
    Load batch experiment result for one player case
    """
    def __init__(self, batch_name, exp_suffix):
        self.batch_name = batch_name
        self.exp_suffix = exp_suffix
        self.data = self.get_statistical_dataframe()
        self.info = self.get_batch_information()
        self.visualizer = None
    
    def get_statistics(self, name="iteration"):
        return self.data[name].describe()
    
    def plot_histogram(self, name="iteration"):
        self.data["iteration"].hist(bins = 20)
    
    def get_batch_information(self):
        if not os.path.exists("result/" + self.batch_name):
            raise ValueError("Batch does not exist: " + self.batch_name)

        list_of_experiments = sorted(
            [dir for dir in os.listdir("result/" + self.batch_name) if self.exp_suffix in dir]
        )

        exp = list_of_experiments[0]
        log_path = os.path.join("result", self.batch_name, exp, "logs", "{}.pkl".format(self.exp_suffix))

        try:
            with open(log_path, "rb") as log:
                data = pickle.load(log)
        except EOFError:
            print("\t\t\t>> Dropping exp {} due to EOFError, please manually check the file".format(exp))
            
        return {
            "g_params": data["g_params"],
            "l_params": data["l_params"]
        }
        
    def get_convergence_rate(self):
        is_converged = self.data["is_converged"]
        convergence_rate = self.data[self.data["is_converged"] == True].shape[0]/self.data.shape[0]
        print("\t>> Convergence rate: {:.3f} ({} of {} runs)".format(
                convergence_rate, len([i for i in is_converged if i is True]), len(is_converged)
            )
        ) 
        return convergence_rate
    
    def get_statistical_dataframe(self):
        is_converged, start_traj, end_traj, terminal_cost, \
        iteration, exp_list, first_negative_cost, value_func, \
        func_key_list, first_t_star, all_traj \
            = self.get_experiment_statistics()
        
        df = pd.DataFrame(
            {
                "is_converged": is_converged, 
                "start_traj": start_traj, 
                "end_traj": end_traj,
                "terminal_cost": terminal_cost, 
                "iteration": iteration,
                "experiment": exp_list,
                "first_negative_cost": first_negative_cost,
                "value_func": value_func,
                "func_key_list": func_key_list,
                "first_t_star": first_t_star,
                "all_traj": all_traj
            }
        )
        
        return df
    
    def get_success_run(self):
        return self.data[self.data["is_converged"]==True]
    
    def visualize_all_runs(self, **kwargs):
        if "data" in kwargs.keys():
            print("\t\t>> Replace all data with filtered data passed")
            data = kwargs["data"]
        else:
            data = self.data

        if "savefig" in kwargs.keys():
            if kwargs["savefig"]:
                summary_dir = os.path.join("result/" + self.batch_name, "summary")
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                print("\t\t>> Save generated image at: {}".format(summary_dir))
            else:
                summary_dir = None
        else:
            summary_dir = None    
    
        if "style" in kwargs.keys():
            style = kwargs["style"]
        else:
            style = "-b"
        
        if "overlay" not in kwargs.keys() or ("overlay" in kwargs.keys() and kwargs["overlay"] is False):
            overlay = False
        else:
            overlay = True

        if "suffix" in kwargs.keys():
            suffix = kwargs["suffix"]
        else:
            suffix = None
        
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
        else:
            alpha = 0.4

        if not overlay:
            self.visualizer = Visualizer(
                [(0, 1)],
                [ProximityCost(self.info["l_params"][0]["car"], self.info["g_params"][0]["car"]), ObstacleDistCost(self.info["g_params"][0]["car"])],
                [style, ".-r", ".-g"],
                1,
                False,
                plot_lims=[-30, 50, 0, 75],
                draw_cars = False
            )
        else:
            self.visualizer._player_linestyles = [style, ".-r", ".-g"]

        for i in range(data.shape[0]):
            xs = data.iloc[i]["end_traj"]
            self.visualizer.add_trajectory(None, {"xs": xs})
            plt.scatter(np.array(xs)[[0]][0][0], np.array(xs)[[0]][0][1], color="firebrick", zorder=10)
            plt.scatter(np.array(xs)[[-1]][0][0], np.array(xs)[[-1]][0][1], color="aqua", zorder=10, alpha = 0.4)
            # self.visualizer.draw_real_car(0, np.array(xs[i][-1])[[0]])
            
            self.visualizer.plot(alpha=alpha, base_size=10.0)
        
        if not overlay:
            if summary_dir is not None:
                if suffix is not None:
                    plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary_{}.svg".format(suffix), format="svg")
                    plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary_{}.png".format(suffix), format="png")
                else:
                    plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary.svg", format="svg")
                    plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary.png", format="png")
            plt.show()
    
    def savefig(self, **kwargs):
        if "suffix" in kwargs.keys():
            suffix = kwargs["suffix"]
        else: 
            suffix = None

        summary_dir = os.path.join("result/" + self.batch_name, "summary")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        print("\t\t>> Save generated image at: {}".format(summary_dir))
        if summary_dir is not None:
            if suffix is not None:
                plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary_{}.svg".format(suffix), format="svg")
                plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary_{}.png".format(suffix), format="png")
            else:
                plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary.svg", format="svg")
                plt.savefig(summary_dir + "/" + self.exp_suffix + "_summary.png", format="png")
        plt.show()
    
    def get_experiment_statistics(self):
        """
        Get necessary data from log file, clean data and return
        """        
        if not os.path.exists("result/" + self.batch_name):
            raise ValueError("Batch does not exist: " + self.batch_name)

        list_of_experiments = sorted(
            [dir for dir in os.listdir("result/" + self.batch_name) if self.exp_suffix in dir]
        )

        is_converged = []
        start_traj = []
        end_traj = []
        terminal_cost = []
        iteration = []
        experiment_list = []
        first_negative_cost = []
        value_func = []
        func_key_list = []
        first_t_star = []
        all_traj = []

        for exp in list_of_experiments:
            log_path = os.path.join("result", self.batch_name, exp, "logs", "{}.pkl".format(self.exp_suffix))

            try:
                with open(log_path, "rb") as log:
                    data = pickle.load(log)
            except EOFError:
                print("\t\t\t>> Dropping exp {} due to EOFError, please manually check the file".format(exp))
                continue

            experiment_list.append(exp)

            # get is_converged data
            if "is_converged" in data.keys():
                if data["is_converged"][0]:
                    is_converged.append(True)
                else:
                    is_converged.append(False)
            else:
                is_converged.append(False)

            # get state data
            if "xs" in data.keys():
                start_traj.append(data["xs"][0])
                end_traj.append(data["xs"][-1])
                all_traj.append(data["xs"])
            else:
                raise ValueError("Cannot find state data")

            # get terminal cost
            if "total_costs" in data.keys():
                terminal_cost.append(min(data["total_costs"]))
                negative_cost_indices = np.array(np.where(np.array(data["total_costs"]) < 0)).flatten()
                try:
                    first_negative_cost.append(negative_cost_indices[0])
                except IndexError:
                    first_negative_cost.append(-1)
            else:
                raise ValueError("Cannot find total costs")
            
            # get value func
            if "value_func_plus" in data.keys():
                value_func.append(data["value_func_plus"][-1])
            else:
                raise ValueError("Cannot find value function")

            # get func_key_list
            if "func_array" in data.keys():
                func_key_list.append(data["func_array"][-1])
            else:
                raise ValueError("Cannot find func_array")

            # get first_t_star
            if "first_t_star" in data.keys():
                first_t_star.append(data["first_t_star"][-1])
            else:
                raise ValueError("Cannot find first t star")

            # get iteration information
            if "iteration" in data.keys():
                iteration.append(max(data["iteration"]))
            else:
                raise ValueError("Cannot find iteration information")

        return is_converged, start_traj, end_traj, terminal_cost, iteration, experiment_list \
            ,first_negative_cost, value_func, func_key_list, first_t_star, all_traj
