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
# Iterative LQ solver.
#
################################################################################

import numpy as np
import math as m
import torch
import matplotlib.pyplot as plt
import os
from collections import deque

from cost.maneuver_penalty import ManeuverPenalty
from player_cost.player_cost import PlayerCost
from cost.proximity_cost import ProximityToBlockCost
from cost.pedestrian_proximity_to_block_cost import PedestrianProximityToBlockCost
from cost.collision_penalty import CollisionPenalty
from cost.pedestrian_collision_penalty import PedestrianToCarCollisionPenalty
from cost.road_rules_penalty import RoadRulesPenalty
from solve_lq_game.solve_lq_game import solve_lq_game
import time
timestr = time.strftime("%Y-%m-%d-%H_%M")

class ILQSolver(object):
    def __init__(self,
                 dynamics,
                 player_costs,
                 x0,
                 Ps,
                 alphas,
                 alpha_scaling= 1.0, # 0.01,
                 reference_deviation_weight=None,
                 logger=None,
                 visualizer=None,
                 u_constraints=None,
                 config=None):
        """
        Initialize from dynamics, player costs, current state, and initial
        guesses for control strategies for both players.

        :param dynamics: two-player dynamical system
        :type dynamics: TwoPlayerDynamicalSystem
        :param player_costs: list of cost functions for all players
        :type player_costs: [PlayerCost]
        :param x0: initial state
        :type x0: np.array
        :param Ps: list of lists of feedback gains (1 list per player)
        :type Ps: [[np.array]]
        :param alphas: list of lists of feedforward terms (1 list per player)
        :type alphas: [[np.array]]
        :param alpha_scaling: step size on the alpha
        :type alpha_scaling: float
        :param reference_deviation_weight: weight on reference deviation cost
        :type reference_deviation_weight: None or float
        :param logger: logging utility
        :type logger: Logger
        :param visualizer: optional visualizer
        :type visualizer: Visualizer
        :param u_constraints: list of constraints on controls
        :type u_constraints: [Constraint]
        """
        self._dynamics = dynamics
        #self._player_costs = player_costs
        self._x0 = x0
        self._Ps = Ps
        self._ns = None
        self._alphas = alphas
        self._u_constraints = u_constraints
        self._horizon = len(Ps[0])
        self._num_players = len(player_costs)
        self.exp_info = config["experiment"]
        self.g_params = config["g_params"]
        self.time_consistency = config["args"].time_consistency
        self.max_steps = config["args"].max_steps
        self.is_batch_run = config["args"].batch_run

        self.plot = config["args"].plot
        self.log = config["args"].log
        self.vel_plot = config["args"].vel_plot
        self.ctl_plot = config["args"].ctl_plot
    
        self.config = config["args"]
        self._player_costs = player_costs

        # Current and previous operating points (states/controls) for use
        # in checking convergence.
        self._last_operating_point = None
        self._current_operating_point = None

        # Fixed step size for the linesearch.
        self._alpha_scaling = alpha_scaling

        # Reference deviation cost weight.
        self._reference_deviation_weight = reference_deviation_weight

        # Set up visualizer.
        self._visualizer = visualizer
        self._logger = logger

        self.linesearch = config["args"].linesearch
        self.linesearch_type = config["args"].linesearch_type

        # Log some of the paramters.
        if self._logger is not None and self.log:
            self._logger.log("horizon", self._horizon)
            self._logger.log("x0", self._x0)
            self._logger.log("config", self.config)
            self._logger.log("g_params", self.g_params)
            self._logger.log("l_params", self.l_params)
            self._logger.log("exp_info", self.exp_info)

    def run(self):
        """ Run the algorithm for the specified parameters. """
        iteration = 0
        # Trying to store stuff in order to plot cost
        store_total_cost = []
        iteration_store = []
        store_freq = 10
        
        while not self._is_converged() and (self.max_steps is not None and iteration < self.max_steps):
            # # (1) Compute current operating point and update last one.
            xs, us = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, us)
            
            # (2) Linearize about this operating point. Make sure to
            # stack appropriately since we will concatenate state vectors
            # but not control vectors, so that
            #    ``` x_{k+1} - xs_k = A_k (x_k - xs_k) +
            #          sum_i Bi_k (ui_k - uis_k) ```
            As = []
            Bs = [[] for ii in range(self._num_players)]
            for k in range(self._horizon):
                A, B = self._dynamics.linearize_discrete(
                    xs[k], [uis[k] for uis in us])
                As.append(A)

                for ii in range(self._num_players):
                    Bs[ii].append(B[ii])
                    

            # (5) Quadraticize costs.
            # Get the hessians and gradients. Hess_x (Q) and grad_x (l) are zero besides at t*
            # Hess_u (R) and grad_u (r) are eps * I and eps * u_t, respectfully, for all [0, T]
            Qs = []
            ls = []
            rs = []
            costs = []
            Rs = []
            calc_deriv_cost = []
            value_func_plus = []
            func_array = []
            func_return_array = []
            total_costs = []
            first_t_stars = []
                        
            for ii in range(self._num_players):           
                Q, l, R, r, costs, total_costss, calc_deriv_cost_, func_array_, func_return_array_, value_func_plus_, first_t_star_ = self._TimeStar(xs, us, ii, first_t_star = True)

                Qs.append(Q[ii])
                ls.append(l[ii])
                rs.append(r[ii])
                
                costs.append(costs[ii])
                calc_deriv_cost.append(calc_deriv_cost_)
                value_func_plus.append(value_func_plus_)
                func_array.append(func_array_)
                func_return_array.append(func_return_array_)
                total_costs.append(total_costss)
                first_t_stars.append(first_t_star_)
                
                Rs.append(R[ii])

            self._first_t_stars = first_t_stars
            self._Qs = Qs
            self._ls = ls
            self._rs = rs
            self._xs = xs
            self._us = us

            # Visualization.
            if self._visualizer is not None:
                traj = {"xs" : xs}
                for ii in range(self._num_players):
                    traj["u%ds" % (ii + 1)] = us[ii]

                self._visualizer.add_trajectory(iteration, traj)
                if self.ctl_plot:
                    self._visualizer.plot_controls(1)
                    plt.pause(0.001)
                    plt.clf()
                    self._visualizer.plot_controls(2)
                    plt.pause(0.001)
                    plt.clf()
                self._visualizer.plot()
                plt.pause(0.001)
                if self.plot:
                    if not os.path.exists(self.exp_info["log_dir"] + "/figures"):
                        os.makedirs(self.exp_info["log_dir"] + "/figures")
                    plt.savefig(self.exp_info["log_dir"] +'/figures/plot-{}.jpg'.format(iteration)) # Trying to save these plots
                plt.clf()

            # draw velocity and timestar overlay graph for 2 cars
            if self.vel_plot:
                for i in range(2):
                    g_critical_index = np.where(np.array(func_array[i]) == "g_x")[0]
                    l_critical_index = np.where(np.array(func_array[i]) == "l_x")[0]
                    value_critical_index = np.where(np.array(func_array[i]) == "value")[0]
                    gradient_critical_index = np.where(np.array(func_array[i]) != "value")[0]
                    plt.figure(4+i)
                    vel_array = np.array([x[5*i + 4] for x in xs]).flatten()
                    
                    plt.plot(vel_array)
                    plt.scatter(g_critical_index, vel_array[g_critical_index], color = "r")
                    plt.scatter(l_critical_index, vel_array[l_critical_index], color = "g")
                    plt.scatter(value_critical_index, vel_array[value_critical_index], color = "y")

                    name_list = []
                    try:
                        for func in np.array(func_return_array[i])[gradient_critical_index]:
                            try:
                                name_list.append(func.__name__.replace("_",""))
                            except:
                                name_list.append(type(func).__name__.replace("_",""))
                        for j in range(len(gradient_critical_index)):
                            plt.text(gradient_critical_index[j], vel_array[gradient_critical_index][j], name_list[j])
                    except Exception as err:
                        print(err)
                        pass
                    plt.pause(0.01)
                    plt.clf()

            # (6) Compute feedback Nash equilibrium of the resulting LQ game.
            # This is getting put into compute_operating_point to solver
            # for the next trajectory
            Ps, alphas, ns = solve_lq_game(As, Bs, Qs, ls, Rs, rs, calc_deriv_cost, self.time_consistency)

            # (7) Accumulate total costs for all players.
            # This is the total cost for the trajectory we are on now
            #total_costs = [sum(costis).item() for costis in costs]
            print("\rInteration: {}\t\tTotal cost for all players:\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(iteration, total_costs[0], total_costs[1], total_costs[2]), end="")
            self._total_costs = total_costs
            
            #Store total cost at each iteration and the iterations
            store_total_cost.append(total_costs)
            iteration_store.append(iteration)

            # Update the member variables.
            self._Ps = Ps
            self._alphas = alphas
            self._ns = ns

            if self.linesearch:
                if self.linesearch_type == "trust_region":
                    self._alpha_scaling = self._linesearch_trustregion(iteration = iteration, visualize_hallucination=True)
                elif self.linesearch_type == "armijo":
                    self._alpha_scaling = self._linesearch_armijo(iteration = iteration)
                else:
                    self._alpha_scaling = 0.05
            else:
                self._alpha_scaling = 1.0 / ((iteration + 1) * 0.5) ** 0.3
                if self._alpha_scaling < .2:
                    self._alpha_scaling = .2
            iteration += 1

            # Log everything.
            if self._logger is not None and self.log and iteration%store_freq == 0:
                self._logger.log("xs", xs)
                self._logger.log("us", us)
                self._logger.log("total_costs", total_costs)
                self._logger.log("alpha_scaling", self._alpha_scaling)
                self._logger.log("calc_deriv_cost", calc_deriv_cost)
                self._logger.log("value_func_plus", value_func_plus)
                self._logger.log("func_array", func_array)
                self._logger.log("func_return_array", func_return_array)
                self._logger.log("first_t_star", first_t_stars)
                self._logger.log("iteration", iteration)
                self._logger.dump()
        
        if self._is_converged():
            print("\nExperiment converged")
            plt.figure()
            plt.plot(iteration_store, np.array(store_total_cost)[:, 0], color='green', linestyle='dashed',linewidth = 2, marker='o', markerfacecolor='blue', markersize=6)
            plt.plot(iteration_store, np.array(store_total_cost)[:, 1], color='red', linestyle='dashed',linewidth = 2, marker='o', markerfacecolor='blue', markersize=6)
            plt.plot(iteration_store, np.array(store_total_cost)[:, 2], color='blue', linestyle='dashed',linewidth = 2, marker='o', markerfacecolor='blue', markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Total cost')
            plt.title('Total Cost of Trajectory at Each Iteration')
            if self.plot:
                plt.savefig(self.exp_info["figure_dir"] +'total_cost_after_{}_steps.jpg'.format(iteration)) # Trying to save these plots
            if not self.is_batch_run:
                plt.show()
            
            if self._logger is not None and self.log:
                self._logger.log("xs", xs)
                self._logger.log("us", us)
                self._logger.log("total_costs", total_costs)
                self._logger.log("alpha_scaling", self._alpha_scaling)
                self._logger.log("calc_deriv_cost", calc_deriv_cost)
                self._logger.log("value_func_plus", value_func_plus)
                self._logger.log("func_array", func_array)
                self._logger.log("func_return_array", func_return_array)
                self._logger.log("first_t_star", first_t_stars)
                self._logger.log("iteration", iteration)
                self._logger.log("is_converged", True)
                self._logger.dump()
        else:
            if self._logger is not None:
                self._logger.log("is_converged", False)
                self._logger.dump()

    def _compute_operating_point(self):
        """
        Compute current operating point by propagating through dynamics.

        :return: states, controls for all players (list of lists), and
            costs over time (list of lists), i.e. (xs, us, costs)
        :rtype: [np.array], [[np.array]], [[torch.Tensor(1, 1)]]
        """
        xs = [self._x0]
        us = [[] for ii in range(self._num_players)]
        #costs = [[] for ii in range(self._num_players)]

        for k in range(self._horizon):
            # If this is our fist time through, we don't have a trajectory, so
            # set the state and controls at each time-step k to 0. Else, use state and
            # controls
            if self._current_operating_point is not None:
                current_x = self._current_operating_point[0][k]
                current_u = [self._current_operating_point[1][ii][k]
                              for ii in range(self._num_players)]
            else:
                current_x = np.zeros((self._dynamics._x_dim, 1))
                current_u = [np.zeros((ui_dim, 1))
                              for ui_dim in self._dynamics._u_dims]
            
            # This is Eqn. 7 in the ILQGames paper
            # This gets us the control at time-step k for the updated trajectory
            feedback = lambda x, u_ref, x_ref, P, alpha : \
                        u_ref - P @ (x - x_ref) - self._alpha_scaling * alpha
            u = [feedback(xs[k], current_u[ii], current_x,
                          self._Ps[ii][k], self._alphas[ii][k])
                  for ii in range(self._num_players)]

            # Append computed control (u) for the trajectory we're calculating to "us"
            for ii in range(self._num_players):
                us[ii].append(u[ii])


            # Use 4th order Runge-Kutta to propogate system to next time-step k+1
            xs.append(self._dynamics.integrate(xs[k], u))
            
        #print("self._aplha_scaling in compute_operating_point is: ", self._alpha_scaling)
        return xs, us
    
    def _is_converged(self):
        """ Check if the last two operating points are close enough. """
        if self._last_operating_point is None:
            return False

        if True:
            return False

        return True
    
    def get_road_logic_dict(self, road_logic):
        return {
            "left_lane": road_logic[0] == 1, 
            "right_lane": road_logic[1] == 1, 
            "up_lane": road_logic[2] == 1, 
            "down_lane": road_logic[3] == 1, 
            "left_turn": road_logic[4] == 1
        }

    def new_road_rules(self, road_logic, road_rules):
        import copy

        left_lane = road_logic["left_lane"]
        right_lane = road_logic["right_lane"]
        down_lane = road_logic["down_lane"]
        up_lane = road_logic["up_lane"]

        new_road_rules = copy.deepcopy(road_rules)

        if down_lane and not up_lane:
            new_road_rules["y_max"] = road_rules["y_max"] - road_rules["width"]
        elif up_lane and not down_lane:
            new_road_rules["y_min"] = road_rules["y_min"] + road_rules["width"]
        
        if left_lane and not right_lane:
            # Can either go straight down or turn left
            new_road_rules["x_max"] = road_rules["x_max"] - road_rules["width"]
        elif right_lane and not left_lane:
            # Can either go straight up or turn right
            new_road_rules["x_min"] = road_rules["x_min"] + road_rules["width"]

        return new_road_rules
    
    def set_player_cost_derivative(self, func_key_list, l_func_list, g_func_list, k, player_index, calc_deriv_cost, is_t_star):
        if is_t_star:
            if func_key_list[k] == "l_x":
                c1gc = l_func_list[k]
                self._player_costs[player_index].add_cost(c1gc, "x", 1.0)
                calc_deriv_cost.appendleft("True")
                self.calc_deriv_cost = True
            elif func_key_list[k] == "g_x":
                c1gc = g_func_list[k]
                self._player_costs[player_index].add_cost(c1gc, "x", 1.0)
                calc_deriv_cost.appendleft("True")
                self.calc_deriv_cost = True
            else:
                c1gc = g_func_list[k]
                self._player_costs[player_index].add_cost(c1gc, "x", 0.0)
                calc_deriv_cost.appendleft("False")
                self.calc_deriv_cost = False
        else:
            if func_key_list[k] == "l_x":
                c1gc = l_func_list[k]
                self._player_costs[player_index].add_cost(c1gc, "x", 0.0)
                calc_deriv_cost.appendleft("False")
                self.calc_deriv_cost = False
            elif func_key_list[k] == "g_x":
                c1gc = g_func_list[k]
                self._player_costs[player_index].add_cost(c1gc, "x", 0.0)
                calc_deriv_cost.appendleft("False")
                self.calc_deriv_cost = False
            else:
                c1gc = g_func_list[k]
                self._player_costs[player_index].add_cost(c1gc, "x", 0.0)
                calc_deriv_cost.appendleft("False")
                self.calc_deriv_cost = False
        return calc_deriv_cost, c1gc

    def _TimeStar(self, xs, us, player_index, **kwargs):
        """
        kwargs: if "first_t_star" is True, regardless of time consistency, pass back the total_cost = cost of first t_star
        """
        car1_position_indices = (0, 1)
        car2_position_indices = (5, 6)
        ped1_position_indices = (10, 11)
        car1_theta_index = 2
        car2_theta_index = 7
        ped1_theta_index = 12
        
        car1_player_id = 0
        car2_player_id = 1
        ped1_player_id = 2

        Qs = [deque() for i in range(self._num_players)]
        ls = [deque() for i in range(self._num_players)]
        rs = [deque() for i in range(self._num_players)]
        Rs = [[deque() for j in range(self._num_players)] for i in range(self._num_players)]
        
        costs = []
        calc_deriv_cost = deque()
        func_return_array = deque()
        
        l_value_list = np.zeros((self._horizon+1, 1))
        g_value_list = np.zeros((self._horizon+1, 1))
        l_func_list = deque()
        g_func_list = deque()
        value_func_plus = np.zeros((self._horizon+1, 1))

        car_params = {
            "wheelbase": 2.413, 
            "length": 4.267,
            "width": 1.988
        }

        road_rules = {
            "x_min": 2,
            "x_max": 9.4,
            "y_max": 27.4,
            "y_min": 20,
            "width": 3.7
        }

        ped_road_rules = {
            "x_min": 2,
            "x_max": 9.4,
            "y_max": 31,
            "y_min": 29
        }

        collision_r = m.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

        # order of road_logic: left, right, up, down, left_turn: [0, 1]
        g_params = {
            "car1": {
                "position_indices": [car1_position_indices, car2_position_indices, ped1_position_indices],
                "player_id": car1_player_id, 
                "road_rules": road_rules,
                "collision_r": collision_r,
                "car_params": car_params,
                "theta_indices": [car1_theta_index, car2_theta_index, ped1_theta_index],
                "road_logic": [0, 1, 0, 1, 0],
                "goals": [20, 35],
                "phi_index": 3, 
                "vel_index": 4
            },
            "car2": {
                "position_indices": [car1_position_indices, car2_position_indices, ped1_position_indices],
                "player_id": car2_player_id, 
                "road_rules": road_rules,
                "collision_r": collision_r,
                "car_params": car_params,
                "theta_indices": [car1_theta_index, car2_theta_index, ped1_theta_index],
                "road_logic": [1, 0, 0, 1, 0],
                "goals": [20, 0],
                "phi_index": 8,
                "vel_index": 9
            },
            "ped1": {
                "position_indices": [car1_position_indices, car2_position_indices, ped1_position_indices],
                "player_id": ped1_player_id, 
                "road_logic": [1, 1, 1, 1, 0],
                "road_rules": ped_road_rules,
                "goals": [15, 30],
                "car_params": car_params,
                "collision_r": collision_r,
                "theta_indices": [car1_theta_index, car2_theta_index, ped1_theta_index],
            }
        }

        l_functions = {
            0: ProximityToBlockCost(g_params["car1"]),
            1: ProximityToBlockCost(g_params["car2"]),
            2: PedestrianProximityToBlockCost(g_params["ped1"], name="ped_goal")
        }

        g_functions = {
            0: self._CheckMultipleGFunctions_P1,
            1: self._CheckMultipleGFunctions_P2,
            2: self._CheckMultipleGFunctions_P3,
        }
        
        func_key_list = [""] * (self._horizon + 1)
        
        for k in range(self._horizon, -1, -1): # T to 0                
            l_func_list.appendleft(
                l_functions[player_index]
            )
            l_value_list[k] = l_func_list[0](xs[k])

            max_g_func = g_functions[player_index](g_params, xs, k)
            g_func_list.appendleft(max_g_func)
            g_value_list[k] = g_func_list[0](xs[k])
            
            value_function_compare = dict()

            if k == self._horizon:
                # if at T, only get max(l_x, g_x)
                value_function_compare = {
                    "g_x": g_value_list[k],
                    "l_x": l_value_list[k]
                }
                value_func_plus[k] = max(value_function_compare.values())
                func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
            else:
                # else, max(g(k), min(l(k), value(k+1)))
                tmp = {
                    "value": value_func_plus[k+1],
                    "l_x": l_value_list[k],
                }
                value_function_compare = {
                    "g_x": g_value_list[k],
                    min(tmp, key=tmp.get): min(tmp.values())
                }
                value_func_plus[k] = max(value_function_compare.values())
                func_key_list[k] = max(value_function_compare, key = value_function_compare.get)

        if "l_x" in func_key_list:
            first_lx_index = func_key_list.index("l_x")
        else:
            first_lx_index = np.inf

        if "g_x" in func_key_list:
            first_gx_index = func_key_list.index("g_x")
        else:
            first_gx_index = np.inf
        
        first_t_star = min(first_lx_index, first_gx_index)

        for k in range(self._horizon, -1, -1): # T to 0
            self._player_costs[player_index] = PlayerCost(**vars(self.config))
            if not self.time_consistency:
                if k == first_t_star:
                    calc_deriv_cost, c1gc = self.set_player_cost_derivative(func_key_list, l_func_list, g_func_list, k, player_index, calc_deriv_cost, is_t_star=True)
                else:
                    calc_deriv_cost, c1gc = self.set_player_cost_derivative(func_key_list, l_func_list, g_func_list, k, player_index, calc_deriv_cost, is_t_star=False)
            else:
                calc_deriv_cost, c1gc = self.set_player_cost_derivative(func_key_list, l_func_list, g_func_list, k, player_index, calc_deriv_cost, is_t_star=True)
            
            func_return_array.appendleft(c1gc)

            if k == self._horizon:
                _, r, l, Q, R = self._player_costs[player_index].quadraticize(
                    xs[k], np.zeros((self._num_players, self._horizon, self._num_players, 1)), k, self.calc_deriv_cost, player_index)
            else:
                _, r, l, Q, R = self._player_costs[player_index].quadraticize(
                    xs[k], [uis[k] for uis in us], k, self.calc_deriv_cost, player_index)

            Qs[player_index].appendleft(Q)
            ls[player_index].appendleft(l)
            rs[player_index].appendleft(r)

            for i in range(self._num_players):
                Rs[player_index][i].appendleft(R[i])
                    
            costs.append(
                self._player_costs[player_index](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self.calc_deriv_cost
                )
            )

        if "first_t_star" in kwargs.keys():
            if kwargs["first_t_star"]:
                total_costs = costs[self._horizon - first_t_star].detach().numpy().flatten()[0]
                return Qs, ls, Rs, rs, costs, total_costs, calc_deriv_cost, func_key_list, func_return_array, value_func_plus, first_t_star
            else:
                total_costs = max(costs).detach().numpy().flatten()[0]
                return Qs, ls, Rs, rs, costs, total_costs, calc_deriv_cost, func_key_list, func_return_array, value_func_plus, first_t_star

        if self.time_consistency:
            total_costs = max(costs).detach().numpy().flatten()[0]
        else:
            total_costs = costs[self._horizon - first_t_star].detach().numpy().flatten()[0]
        
        return Qs, ls, Rs, rs, costs, total_costs, calc_deriv_cost, func_key_list, func_return_array, value_func_plus, first_t_star
    
    def _CheckMultipleGFunctions_P1(self, g_params, xs, k):
        max_func = dict()
        
        max_val, func_of_max_val = CollisionPenalty(g_params["car1"])(xs[k])
        max_func[func_of_max_val] = max_val
        
        max_val, func_of_max_val = RoadRulesPenalty(g_params["car1"])(xs[k])
        max_func[func_of_max_val] = max_val

        max_val, func_of_max_val = ManeuverPenalty(g_params["car1"])(xs[k])
        max_func[func_of_max_val] = max_val

        return max(max_func, key=max_func.get)

    def _CheckMultipleGFunctions_P2(self, g_params, xs, k):
        max_func = dict()
        
        max_val, func_of_max_val = CollisionPenalty(g_params["car2"])(xs[k])
        max_func[func_of_max_val] = max_val
        
        max_val, func_of_max_val = RoadRulesPenalty(g_params["car2"])(xs[k])
        max_func[func_of_max_val] = max_val

        max_val, func_of_max_val = ManeuverPenalty(g_params["car2"])(xs[k])
        max_func[func_of_max_val] = max_val

        return max(max_func, key=max_func.get)

    def _CheckMultipleGFunctions_P3(self, g_params, xs, k):
        max_func = dict()
        
        max_val, func_of_max_val = PedestrianToCarCollisionPenalty(g_params["car1"])(xs[k])
        max_func[func_of_max_val] = max_val

        max_val, func_of_max_val = PedestrianToCarCollisionPenalty(g_params["car2"])(xs[k])
        max_func[func_of_max_val] = max_val
        
        return max(max_func, key=max_func.get)

    def _rollout(self, xs, us, player_index, **kwargs):
        """
        kwargs: if "first_t_star" is True, regardless of time consistency, pass back the total_cost = cost of first t_star
        """
        car1_position_indices = (0, 1)
        car2_position_indices = (5, 6)
        ped1_position_indices = (10, 11)
        car1_theta_index = 2
        car2_theta_index = 7
        ped1_theta_index = 12
        
        car1_player_id = 0
        car2_player_id = 1
        ped1_player_id = 2
        
        costs = []
        
        l_value_list = np.zeros((self._horizon+1, 1))
        g_value_list = np.zeros((self._horizon+1, 1))
        l_func_list = deque()
        g_func_list = deque()
        value_func_plus = np.zeros((self._horizon+1, 1))

        car_params = {
            "wheelbase": 2.413, 
            "length": 4.267,
            "width": 1.988
        }

        road_rules = {
            "x_min": 2,
            "x_max": 9.4,
            "y_max": 27.4,
            "y_min": 20,
            "width": 3.7
        }

        ped_road_rules = {
            "x_min": 2,
            "x_max": 9.4,
            "y_max": 31,
            "y_min": 29
        }

        collision_r = m.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

        # order of road_logic: left, right, up, down, left_turn: [0, 1]
        g_params = {
            "car1": {
                "position_indices": [car1_position_indices, car2_position_indices, ped1_position_indices],
                "player_id": car1_player_id, 
                "road_rules": road_rules,
                "collision_r": collision_r,
                "car_params": car_params,
                "theta_indices": [car1_theta_index, car2_theta_index, ped1_theta_index],
                "road_logic": [0, 1, 0, 1, 0],
                "goals": [20, 35],
                "phi_index": 3, 
                "vel_index": 4
            },
            "car2": {
                "position_indices": [car1_position_indices, car2_position_indices, ped1_position_indices],
                "player_id": car2_player_id, 
                "road_rules": road_rules,
                "collision_r": collision_r,
                "car_params": car_params,
                "theta_indices": [car1_theta_index, car2_theta_index, ped1_theta_index],
                "road_logic": [1, 0, 0, 1, 0],
                "goals": [20, 0],
                "phi_index": 8,
                "vel_index": 9
            },
            "ped1": {
                "position_indices": [car1_position_indices, car2_position_indices, ped1_position_indices],
                "player_id": ped1_player_id, 
                "road_logic": [1, 1, 1, 1, 0],
                "road_rules": ped_road_rules,
                "goals": [15, 30],
                "car_params": car_params,
                "collision_r": collision_r,
                "theta_indices": [car1_theta_index, car2_theta_index, ped1_theta_index],
            }
        }

        l_functions = {
            0: ProximityToBlockCost(g_params["car1"]),
            1: ProximityToBlockCost(g_params["car2"]),
            2: PedestrianProximityToBlockCost(g_params["ped1"], name="ped_goal")
        }

        g_functions = {
            0: self._CheckMultipleGFunctions_P1,
            1: self._CheckMultipleGFunctions_P2,
            2: self._CheckMultipleGFunctions_P3,
        }
        
        func_key_list = [""] * (self._horizon + 1)
        
        for k in range(self._horizon, -1, -1): # T to 0                
            l_func_list.appendleft(
                l_functions[player_index]
            )
            l_value_list[k] = l_func_list[0](xs[k])

            max_g_func = g_functions[player_index](g_params, xs, k)
            g_func_list.appendleft(max_g_func)
            g_value_list[k] = g_func_list[0](xs[k])
            
            value_function_compare = dict()

            if k == self._horizon:
                # if at T, only get max(l_x, g_x)
                value_function_compare = {
                    "g_x": g_value_list[k],
                    "l_x": l_value_list[k]
                }
                value_func_plus[k] = max(value_function_compare.values())
                func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
            else:
                # else, max(g(k), min(l(k), value(k+1)))
                tmp = {
                    "value": value_func_plus[k+1],
                    "l_x": l_value_list[k],
                }
                value_function_compare = {
                    "g_x": g_value_list[k],
                    min(tmp, key=tmp.get): min(tmp.values())
                }
                value_func_plus[k] = max(value_function_compare.values())
                func_key_list[k] = max(value_function_compare, key = value_function_compare.get)

        if "l_x" in func_key_list:
            first_lx_index = func_key_list.index("l_x")
        else:
            first_lx_index = np.inf

        if "g_x" in func_key_list:
            first_gx_index = func_key_list.index("g_x")
        else:
            first_gx_index = np.inf
        
        first_t_star = min(first_lx_index, first_gx_index)

        for k in range(self._horizon, -1, -1): # T to 0
            self._player_costs[player_index] = PlayerCost(**vars(self.config))
            if not self.time_consistency:
                if k == first_t_star:
                    _, c1gc = self.set_player_cost_derivative(func_key_list, l_func_list, g_func_list, k, player_index, deque(), is_t_star=True)
                else:
                    _, c1gc = self.set_player_cost_derivative(func_key_list, l_func_list, g_func_list, k, player_index, deque(), is_t_star=False)
            else:
                _, c1gc = self.set_player_cost_derivative(func_key_list, l_func_list, g_func_list, k, player_index, deque(), is_t_star=True)
                    
            costs.append(
                self._player_costs[player_index](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self.calc_deriv_cost
                )
            )

        if "first_t_star" in kwargs.keys():
            if kwargs["first_t_star"]:
                return first_t_star, costs[self._horizon - first_t_star].detach().numpy().flatten()[0]
            else:
                return first_t_star, max(costs).detach().numpy().flatten()[0]

        if self.time_consistency:
            total_costs = max(costs).detach().numpy().flatten()[0]
        else:
            total_costs = costs[self._horizon - first_t_star].detach().numpy().flatten()[0]
        
        return first_t_star, total_costs
    
    def _linesearch_residual(self, beta = 0.9, iteration = None):
        """ Linesearch for both players separately. """
        """
        x -> us
        p -> rs
        may need xs to compute trajectory
        Line search needs c and tau (c = tau = 0.5 default)
        m -> local slope (calculate in function)
        need compute_operating_point routine
        """        
        
        alpha_converged = False
        alpha = 1.0
        expected_improvement = np.zeros(self._num_players)
        total_costs_new = np.zeros(self._num_players)
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            
            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            for ii in range(self._num_players):
                t_star, total_cost_new = self._TimeStarRollout(xs, us, ii)

                expected_rate = self._ns[ii][0]
                expected_improvement[ii] = expected_rate * alpha

                total_costs_new[ii] = total_cost_new
            
            expected_improvement = np.zeros(self._num_players)
            if max(total_costs_new - self._total_costs - expected_improvement) < 0:
                alpha_converged = True
                return alpha
            else:
                alpha = beta * alpha
                if iteration is not None:
                    if alpha < 1.0/(iteration+1) ** 0.5:
                        return alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small") 
            
        self._alpha_scaling = alpha
        return alpha

    def _linesearch_trustregion(self, beta = 0.9, iteration = None, margin=5.0, visualize_hallucination = False):
        """ Linesearch using trust region. """
        """
        beta (float) -> discounted term
        iteration (int) -> current iteration
        """
        
        alpha_converged = False
        alpha = 1.0
        error = old_error = np.zeros(self._num_players).astype(np.float16)
        total_costs_new = np.zeros(self._num_players)
        delta_costs_quadratic_approx = np.zeros(self._num_players)
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha

            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here

            # Visualize hallucinated traj
            if visualize_hallucination:
                plt.figure(1)
                self._visualizer.plot()
                plt.plot([x[0, 0] for x in xs], [x[1, 0] for x in xs],
                    "-g",
                    alpha = 0.4,
                    linewidth = 2,
                    zorder=10
                )
                plt.plot([x[5, 0] for x in xs], [x[6, 0] for x in xs],
                    "-r",
                    alpha = 0.4,
                    linewidth = 2,
                    zorder=10
                )
                plt.plot([x[10, 0] for x in xs], [x[11, 0] for x in xs],
                    "-b",
                    alpha = 0.4,
                    linewidth = 2,
                    zorder=10
                )
                plt.pause(0.001)
                plt.clf()
            
            for i in range(self._num_players):
                new_t_star, total_cost_new = self._rollout(xs, us, i, first_t_star=True)

                total_costs_new[i] = total_cost_new

                old_t_star = self._first_t_stars[i]
                Q = self._Qs[i][old_t_star]
                q = self._ls[i][old_t_star]

                x_diff = [(np.array(x_new) - np.array(x_old)) for x_new, x_old in zip(np.array(xs)[old_t_star,:,:], np.array(self._current_operating_point[0])[old_t_star,:,:])]
                delta_cost_quadratic_approx = 0.5 * (np.transpose(x_diff) @ Q + 2 * np.transpose(q)) @ x_diff

                delta_costs_quadratic_approx[i] = delta_cost_quadratic_approx
            
            traj_diff = max([np.linalg.norm(np.array(x_new) - np.array(x_old)) for x_new, x_old in zip(np.array(xs)[:,[0, 1, 5, 6, 10, 11],:], np.array(self._current_operating_point[0])[:,[0, 1, 5, 6, 10, 11],:])])

            delta_cost_quadratic_actual = total_costs_new - self._total_costs
            error = np.array(delta_cost_quadratic_approx - delta_cost_quadratic_actual).astype(np.float16)

            if traj_diff < margin:
                if np.array_equal(old_error, error):
                    alpha_converged = True
                else:
                    old_error = error
                    if abs(error[0, 0]) < 1.2 and abs(error[0, 0]) > 0.8:
                        margin = margin * 1.5
                        alpha = 1.0
                        alpha_converged = False
                    elif abs(error[0, 0]) >= 1.2:
                        margin = margin * 0.5
                        alpha = 1.0
                        alpha_converged = False
                    else:
                        alpha_converged = True
            else:
                alpha = beta * alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small")

            # print("Est cost: {:.5f}\tNew cost: {:.5f}\tAlpha: {:.5f}\tMargin: {:.5f}\tdelta cost: {:.5f}\tTraj diff: {:.5f}".format((delta_cost_quadratic_approx + self._total_costs).flatten()[0], total_costs_new[0], alpha, margin, delta_cost_quadratic_approx.flatten()[0], traj_diff))
        
        self._alpha_scaling = alpha
        return alpha

    def _linesearch_residual(self, beta = 0.9, iteration = None):
        """ Linesearch for both players separately. """
        """
        x -> us
        p -> rs
        may need xs to compute trajectory
        Line search needs c and tau (c = tau = 0.5 default)
        m -> local slope (calculate in function)
        need compute_operating_point routine
        """        
        
        alpha_converged = False
        alpha = 1.0
        expected_improvement = np.zeros(self._num_players)
        total_costs_new = np.zeros(self._num_players)
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            
            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            for ii in range(self._num_players):
                t_star, total_cost_new = self._rollout(xs, us, ii)

                expected_rate = self._ns[ii][0]
                expected_improvement[ii] = expected_rate * alpha

                total_costs_new[ii] = total_cost_new
            
            expected_improvement = np.zeros(self._num_players)
            if max(total_costs_new - self._total_costs - expected_improvement) < 0:
                alpha_converged = True
                return alpha
            else:
                alpha = beta * alpha
                if iteration is not None:
                    if alpha < 1.0/(iteration+1) ** 0.5:
                        return alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small") 
            
        self._alpha_scaling = alpha
        return alpha

    def _linesearch_armijo(self, beta=0.9, iteration = None):
        """ Linesearch for both players separately. """
        """
        x -> us
        p -> rs
        may need xs to compute trajectory
        Line search needs c and tau (c = tau = 0.5 default)
        m -> local slope (calculate in function)
        need compute_operating_point routine
        """        
        
        alpha_converged = False
        alpha = 1.0
        total_costs_new = np.zeros(self._num_players)
        ts = np.zeros(self._num_players)
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            for ii in range(self._num_players):
                t_star, total_cost_new = self._rollout(xs, us, ii)
                total_costs_new[ii] = total_cost_new

                if t_star < self._horizon:
                    # Calculate p (delta_u in our case)
                    delta_u = -self._Ps[ii][t_star] @ (xs[t_star] - self._current_operating_point[0][t_star]) - self._alphas[ii][t_star]
                    grad_cost_u = self._rs[ii][t_star]
                    ts[ii] = -0.5 * grad_cost_u[ii] @ delta_u
            
            if max(total_costs_new  + ts * alpha - self._total_costs) <= 0:
                alpha_converged = True
                return alpha
            else:
                alpha = beta * alpha
                if iteration is not None:
                    if alpha < 1.0/(iteration+1) ** 0.5:
                        return alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small")
        
        self._alpha_scaling = alpha
        return alpha