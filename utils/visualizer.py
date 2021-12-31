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
# Fancy visualization class.
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation, markers
import os
from matplotlib.transforms import Affine2D

from resource.car_5d import Car5Dv2

class Visualizer(object):
    def __init__(self,
                 position_indices,
                 renderable_costs,
                 player_linestyles,
                 show_last_k=1,
                 fade_old=False,
                 plot_lims=None,
                 figure_number=1,
                 **kwargs):
        """
        Construct from list of position indices and renderable cost functions.

        :param position_indices: list of tuples of position indices (1/player)
        :type position_indices: [(uint, uint)]
        :param renderable_costs: list of cost functions that support rendering
        :type renderable_costs: [Cost]
        :param player_linestyles: list of line styles (1 per player, e.g. ".-r")
        :type player_colors: [string]
        :param show_last_k: how many of last trajectories to plot (-1 shows all)
        :type show_last_k: int
        :param fade_old: flag for fading older trajectories
        :type fade_old: bool
        :param plot_lims: plot limits [xlim_low, xlim_high, ylim_low, ylim_high]
        :type plot_lims: [float, float, float, float]
        :param figure_number: which figure number to operate on
        :type figure_number: uint
        """
        self._position_indices = position_indices
        self._renderable_costs = renderable_costs
        self._player_linestyles = player_linestyles
        self._show_last_k = show_last_k
        self._fade_old = fade_old
        self._figure_number = figure_number
        self._plot_lims = plot_lims
        self._num_players = len(position_indices)

        # Store history as list of trajectories.
        # Each trajectory is a dictionary of lists of states and controls.
        self._iterations = []
        self._history = []

        self._draw_roads = False
        self._draw_cars = False
        self._adversarial = False
        self._draw_human = False

        if "draw_roads" in kwargs.keys():
            self._draw_roads = kwargs["draw_roads"]
        if "draw_cars" in kwargs.keys():
            self._draw_cars = kwargs["draw_cars"]
        if "adversarial" in kwargs.keys():
            self._adversarial = kwargs["adversarial"]
        if "t_react" in kwargs.keys():
            self._t_react = kwargs["t_react"]
        if "draw_human" in kwargs.keys():
            self._draw_human = kwargs["draw_human"]

    def add_trajectory(self, iteration, traj):
        """
        Add a new trajectory to the history.

        :param iteration: which iteration is this
        :type iteration: uint
        :param traj: trajectory
        :type traj: {"xs": [np.array], "u1s": [np.array], "u2s": [np.array]}
        """
        self._iterations.append(iteration)
        self._history.append(traj)

    def linewidth_from_data_units(self, linewidth, axis, reference='y'):
        """
        Convert a linewidth in data units to linewidth in points.

        Parameters
        ----------
        linewidth: float
            Linewidth in data units of the respective reference-axis
        axis: matplotlib axis
            The axis which is used to extract the relevant transformation
            data (data limits and size must not change afterwards)
        reference: string
            The axis that is taken as a reference for the data width.
            Possible values: 'x' and 'y'. Defaults to 'y'.

        Returns
        -------
        linewidth: float
            Linewidth in points
        """
        fig = plt.gcf()
        if reference == 'x':
            length = fig.bbox_inches.width * axis.get_position().width
            value_range = np.diff(axis.get_xlim())
        elif reference == 'y':
            length = fig.bbox_inches.height * axis.get_position().height
            value_range = np.diff(axis.get_ylim())
        # Convert length to points
        length *= 72
        # Scale linewidth to value range
        return linewidth * (length / value_range)

    def draw_crosswalk(self, x, y, width, length, number_of_dashes = 5):
        per_length = length * 0.5 / number_of_dashes
        for i in range(number_of_dashes):
            crosswalk = plt.Rectangle(
                [x + (2*i + 0.5)*per_length, y], width = per_length, height = width, color = "white", lw = 0, zorder = 0)
            plt.gca().add_patch(crosswalk)

    def draw_car(self, player_id, car_states):
        car_params = {
            "wheelbase": 2.413, 
            "length": 4.267,
            "width": 1.988
        }

        for i in range(len(car_states)):
            if player_id == 0:
                state = car_states[i][:5].flatten()
                color = "r"
            else:
                state = car_states[i][5:].flatten()
                color = "g"
            plt.gca().set_aspect('equal')
            rotate_deg = state[2]/np.pi * 180
            length = car_params["length"]
            width = car_params["width"]
            wheelbase = car_params["wheelbase"]
            a = 0.5 * (length - wheelbase)

            plt.plot(state[0], state[1], color=color, marker='o', markersize=5, alpha = 0.4)
            if i % 5 == 0:
                rec = plt.Rectangle(state[:2] - np.array([a, 0.5*width]), width = length, height = width, color = color, alpha=(1.0/len(car_states))*i,
                                        transform=Affine2D().rotate_deg_around(*(state[0], state[1]), rotate_deg) + plt.gca().transData)
                plt.gca().add_patch(rec)

    def draw_real_car(self, player_id, car_states, path=None):
        # TODO: change all the constants in the function to car_params
        car_params = {
            "wheelbase": 2.413, 
            "length": 4.267,
            "width": 1.988
        }
        
        for i in range(len(car_states)):
            if player_id == 0:
                state = car_states[i][:5].flatten()
                color = "r"
                path = "visual_components/delorean.png" if path is None else path
            else:
                state = car_states[i][5:].flatten()
                color = "g"
                path = "visual_components/car_robot_r.png" if path is None else path

            transform_data = Affine2D().rotate_deg_around(*(state[0], state[1]), state[2]/np.pi * 180) + plt.gca().transData
            # plt.plot(state[0], state[1], color=color, marker='o', markersize=5, alpha = 0.4)
            if i % 5 == 0:
                plt.imshow(
                    plt.imread(path, format="png"), 
                    transform = transform_data, 
                    interpolation='none',
                    origin='lower',
                    extent=[state[0] - 0.927, state[0] + 3.34, state[1] - 0.944, state[1] + 1.044],
                    alpha = 1.0, 
                    # alpha=(1.0/len(car_states))*i,
                    zorder = 10.0,
                    clip_on=True)

    def draw_real_human(self, states, variation=0):
        for i in range(len(states)):
            state = states[i][10:].flatten()
            transform_data = Affine2D().rotate_deg_around(*(state[0], state[1]), (state[2] + np.pi * 0.5)/np.pi * 180) + plt.gca().transData
            if i % 5 == 0:
                plt.imshow(
                    plt.imread("visual_components/human-walking-topdown-step{}.png".format(variation), format="png"), 
                    transform = transform_data, 
                    interpolation='none',
                    origin='lower',
                    extent=[state[0] - 1.2, state[0] + 1.2, state[1] + 1.2, state[1] - 1.2],
                    zorder = 10.0,
                    clip_on=True
                )
    
    def draw_road_rules(self, ax, **kwargs):
        if "road_rules" in kwargs.keys():
            road_rules = kwargs["road_rules"]
        else:
            road_rules = {
                "x_min": 2,
                "x_max": 9.4,
                "y_max": 27.4,
                "y_min": 20,
                "width": 3.7
            }

        x_max = 25
        y_max = 40

        grass = plt.Rectangle(
        [-5, 0], width = 30, height = 40, color = "k", lw = 0, zorder = -2, alpha = 0.5)
        plt.gca().add_patch(grass)  
        
        # plot road rules
        x_center = road_rules["x_min"] + 0.5 * (road_rules["x_max"] - road_rules["x_min"])
        y_center = road_rules["y_min"] + 0.5 * (road_rules["y_max"] - road_rules["y_min"])
        road = plt.Rectangle(
        [road_rules["x_min"], 0], width = road_rules["x_max"] - road_rules["x_min"], height = y_max, color = "darkgray", lw = 0, zorder = -2)
        plt.gca().add_patch(road)
        road = plt.Rectangle(
            [road_rules["x_max"], road_rules["y_min"]], width = x_max, height = road_rules["y_max"] - road_rules["y_min"], color = "darkgray", lw = 0, zorder = -2)
        plt.gca().add_patch(road)

        crosswalk_width = 3
        crosswalk_length = road_rules["x_max"] - road_rules["x_min"]
        self.draw_crosswalk(road_rules["x_min"], 30 - crosswalk_width*0.5, crosswalk_width, crosswalk_length)

        ax.plot([road_rules["x_min"], road_rules["x_min"]], [0, y_max], c="white", linewidth = 2, zorder = -1)
        ax.plot([road_rules["x_max"], road_rules["x_max"]], [0, road_rules["y_min"]], c="white", linewidth = 2, zorder = -1)
        ax.plot([road_rules["x_max"], road_rules["x_max"]], [road_rules["y_max"], y_max], c="white", linewidth = 2, zorder = -1)
        ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_min"], road_rules["y_min"]], c="white", linewidth = 2, zorder = -1)
        ax.plot([road_rules["x_max"], x_max], [road_rules["y_min"], road_rules["y_min"]], c="white", linewidth = 2, zorder = -1)
        ax.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_max"], road_rules["y_max"]], c="white", linewidth = 2, zorder = -1)
        ax.plot([road_rules["x_max"], x_max], [road_rules["y_max"], road_rules["y_max"]], c="white", linewidth = 2, zorder = -1)
        ax.plot([x_center, x_center], [0, y_max], "--", c = 'white', linewidth = 5, dashes=(5, 5), zorder = -1)
        ax.plot([road_rules["x_max"], x_max], [y_center, y_center], "--", c = 'white', linewidth = 5, dashes=(5, 5), zorder = -1)

    def plot(self):
        """ Plot everything. """
        
        # script_dir = os.path.dirname(__file__)
        # results_dir = os.path.join(script_dir, 'results/')
        # sample_file_name = "sample"
        
        # if not os.path.isdir(results_dir):
        #     os.makedirs(results_dir)
        
        ratio = (self._plot_lims[1] - self._plot_lims[0])/(self._plot_lims[3] - self._plot_lims[2])
        plt.figure(self._figure_number, figsize=(ratio*10, 10))
        # plt.rc("text", usetex=True)

        ax = plt.gca()
        plt.axis("off")
        # ax.set_xlabel("$x(t)$")
        # ax.set_ylabel("$y(t)$")

        if self._plot_lims is not None:
            ax.set_xlim(self._plot_lims[0], self._plot_lims[1])
            ax.set_ylim(self._plot_lims[2], self._plot_lims[3])

        ax.set_aspect("equal")

        # Render all costs.
        for cost in self._renderable_costs:
            cost.render(ax)

        # draw roads
        if self._draw_roads:
            self.draw_road_rules(ax)

        # Plot the history of trajectories for each player.
        if self._show_last_k < 0 or self._show_last_k >= len(self._history):
            show_last_k = len(self._history)
        else:
            show_last_k = self._show_last_k

        plotted_iterations = []
        for kk in range(len(self._history) - show_last_k, len(self._history)):
            traj = self._history[kk]
            iteration = self._iterations[kk]
            plotted_iterations.append(iteration)

            for ii in range(self._num_players):
                x_idx, y_idx = self._position_indices[ii]
                xs = [x[x_idx, 0] for x in traj["xs"]]
                ys = [x[y_idx, 0] for x in traj["xs"]]
                if ii == 0:
                    # self.draw_car(ii, traj["xs"])
                    if self._draw_cars:
                        self.draw_real_car(ii, traj["xs"])
                    else:
                        plt.plot(xs, ys,
                            self._player_linestyles[ii],
                            label = "Player {}, iteration {}".format(ii, iteration),
                            # alpha = 0.4,
                            linewidth = 2
                            #  linewidth = self.linewidth_from_data_units(1.988, ax)
                        )
                elif ii == 1:
                    # self.draw_car(ii, traj["xs"])
                    if self._draw_cars:
                        if self._adversarial:
                            self.draw_real_car(ii, traj["xs"][:self._t_react])
                            self.draw_real_car(ii, traj["xs"][self._t_react:], path = "visual_components/car_robot_y.png")
                        else:
                            self.draw_real_car(ii, traj["xs"])
                    else:
                        if self._adversarial:
                            plt.plot(xs[:self._t_react], ys[:self._t_react],
                                ".-y",
                                label = "Player {}, iteration {}".format(ii, iteration),
                                alpha = 0.4,
                                linewidth = 2
                                #  linewidth = self.linewidth_from_data_units(1.988, ax)
                            )
                            plt.plot(xs[self._t_react:], ys[self._t_react:],
                                self._player_linestyles[ii],
                                label = "Player {}, iteration {}".format(ii, iteration),
                                alpha = 0.4,
                                linewidth = 2
                                #  linewidth = self.linewidth_from_data_units(1.988, ax)
                            )
                        else:
                            plt.plot(xs, ys,
                                self._player_linestyles[ii],
                                label = "Player {}, iteration {}".format(ii, iteration),
                                alpha = 0.4,
                                linewidth = 2
                                #  linewidth = self.linewidth_from_data_units(1.988, ax)
                            )
                else:
                    if self._draw_human:
                        self.draw_real_human(traj["xs"])
                    else:
                        plt.plot(xs, ys,
                            self._player_linestyles[ii],
                            label = "Player {}, iteration {}".format(ii, iteration),
                            alpha = 0.4,
                            linewidth = 2,
                            marker='o', markersize = 10
                            #  linewidth = self.linewidth_from_data_units(1.988, ax)
                        )

        # plt.title("ILQ solver solution (iterations {}-{})".format(
        #     plotted_iterations[0], plotted_iterations[-1]))
        
        # plt.savefig(results_dir + sample_file_name) # trying to save figure

    def plot_simplified(self):
        """ Plot everything, simplified """
        
        # script_dir = os.path.dirname(__file__)
        # results_dir = os.path.join(script_dir, 'results/')
        # sample_file_name = "sample"
        
        # if not os.path.isdir(results_dir):
        #     os.makedirs(results_dir)
        
        plt.figure(self._figure_number, figsize=(8, 10))
        ax = plt.gca()

        if self._plot_lims is not None:
            ax.set_xlim(self._plot_lims[0], self._plot_lims[1])
            ax.set_ylim(self._plot_lims[2], self._plot_lims[3])

        ax.set_aspect("equal")

        if self._draw_roads:
            self.draw_road_rules(ax)

        # Plot the history of trajectories for each player.
        if self._show_last_k < 0 or self._show_last_k >= len(self._history):
            show_last_k = len(self._history)
        else:
            show_last_k = self._show_last_k

        plotted_iterations = []
        for kk in range(len(self._history) - show_last_k, len(self._history)):
            traj = self._history[kk]
            iteration = self._iterations[kk]
            plotted_iterations.append(iteration)

            for ii in range(self._num_players):
                x_idx, y_idx = self._position_indices[ii]
                xs = [x[x_idx, 0] for x in traj["xs"]]
                ys = [x[y_idx, 0] for x in traj["xs"]]
                if ii == 0 or ii == 1:
                    plt.plot(
                        xs, ys,
                        linewidth = 2
                    )
                else:
                    plt.plot(xs, ys,
                        self._player_linestyles[ii],
                        linewidth = 2,
                        marker='o', markersize = 10
                    )

        plt.title("ILQ solver solution (iterations {}-{})".format(
            plotted_iterations[0], plotted_iterations[-1]))

    def plot_controls(self, player_number):
        """ Plot control for both players. """
        plt.figure(self._figure_number + player_number)
        uis = "u%ds" % player_number
        plt.plot([ui[0, 0] for ui in self._history[-1][uis]], "*:r", label="u1")
        plt.plot([ui[1, 0] for ui in self._history[-1][uis]], "*:b", label="u2")
        plt.legend()
        plt.title("Controls for Player %d" % player_number)
