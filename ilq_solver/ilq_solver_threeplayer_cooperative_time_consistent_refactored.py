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
from numpy.lib.function_base import gradient
import torch
import matplotlib.pyplot as plt
import os
from scipy.linalg import block_diag
from collections import deque

from torch.autograd import grad
from maneuver_penalty import ManeuverPenalty

from player_cost_reachavoid_timeconsistent import PlayerCost
from proximity_cost_reach_avoid_twoplayer import PedestrianProximityToBlockCost, ProximityToBlockCost
from distance_twoplayer_cost import CollisionPenalty, PedestrianToCarCollisionPenalty
from semiquadratic_polyline_cost_any import RoadRulesPenalty
from solve_lq_game_reachavoid_timeconsistent import solve_lq_game
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
                 u_constraints=None):
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

        # Log some of the paramters.
        if self._logger is not None:
            self._logger.log("alpha_scaling", self._alpha_scaling)
            self._logger.log("horizon", self._horizon)
            self._logger.log("x0", self._x0)

    def run(self):
        """ Run the algorithm for the specified parameters. """
        iteration = 0
        # Trying to store stuff in order to plot cost
        store_total_cost = []
        iteration_store = []
        store_freq = 5
        
        while not self._is_converged():
            # # (1) Compute current operating point and update last one.
            xs, us = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, us)

            # print(max([x[4] for x in xs]))
            # print(max([x[9] for x in xs]))
            # time.sleep(0.2)
            
            if iteration%store_freq == 0:
                xs_store = [xs_i.flatten() for xs_i in xs]
                #print(xs_store[0])
                #print(len(xs_store))
                #np.savetxt('horizontal_treact20_'+str(iteration)+'.out', np.array(xs_store), delimiter = ',')

                np.savetxt('logs/three_player/threeplayer_intersection_'+str(iteration)+'.txt', np.array(xs_store), delimiter = ',')
            

            # Visualization.
            if self._visualizer is not None:
                traj = {"xs" : xs}
                for ii in range(self._num_players):
                    traj["u%ds" % (ii + 1)] = us[ii]

                self._visualizer.add_trajectory(iteration, traj)
                # self._visualizer.plot_controls(1)
                # plt.pause(0.001)
                # plt.clf()
                # self._visualizer.plot_controls(2)
                # plt.pause(0.001)
                # plt.clf()
                self._visualizer.plot()
                plt.pause(0.001)
                if not os.path.exists("image_outputs_{}".format(timestr)):
                    os.makedirs("image_outputs_{}".format(timestr))
                plt.savefig('image_outputs_{}/plot-{}.jpg'.format(timestr, iteration)) # Trying to save these plots
                plt.clf()
            
            # print("plot is shown above")
            # print("self._num_players is: ", self._num_players)                

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
            func_array = []
            func_return_array = []
            total_costs = []
                        
            for ii in range(self._num_players):           
                Q, l, R, r, costs, total_costss, calc_deriv_cost_, func_array_, func_return_array_ = self._TimeStar(xs, us, ii)

                Qs.append(Q[ii])
                ls.append(l[ii])
                rs.append(r[ii])
                
                costs.append(costs[ii])
                calc_deriv_cost.append(calc_deriv_cost_)
                func_array.append(func_array_)
                func_return_array.append(func_return_array_)
                total_costs.append(total_costss)
                
                Rs.append(R[ii])
        
            self._rs = rs
            self._xs = xs
            self._us= us

            # draw velocity and timestar overlay graph for 2 cars
            # for i in range(2):
            #     g_critical_index = np.where(np.array(func_array[i]) == "g_x")[0]
            #     l_critical_index = np.where(np.array(func_array[i]) == "l_x")[0]
            #     value_critical_index = np.where(np.array(func_array[i]) == "value")[0]
            #     gradient_critical_index = np.where(np.array(func_array[i]) != "value")[0]
            #     plt.figure(4+i)
            #     vel_array = np.array([x[5*i + 4] for x in xs]).flatten()
                
            #     plt.plot(vel_array)
            #     plt.scatter(g_critical_index, vel_array[g_critical_index], color = "r")
            #     plt.scatter(l_critical_index, vel_array[l_critical_index], color = "g")
            #     plt.scatter(value_critical_index, vel_array[value_critical_index], color = "y")

            #     name_list = []
            #     try:
            #         for func in np.array(func_return_array[i])[gradient_critical_index]:
            #             try:
            #                 name_list.append(func.__name__.replace("_",""))
            #             except:
            #                 name_list.append(type(func).__name__.replace("_",""))
            #         for j in range(len(gradient_critical_index)):
            #             plt.text(gradient_critical_index[j], vel_array[gradient_critical_index][j], name_list[j])
            #     except Exception as err:
            #         print(err)
            #         pass
            #     plt.pause(0.01)
            #     plt.clf()

            # (6) Compute feedback Nash equilibrium of the resulting LQ game.
            # This is getting put into compute_operating_point to solver
            # for the next trajectory
            # print(np.array(Qs).shape)
            # input()
            Ps, alphas, ns = solve_lq_game(As, Bs, Qs, ls, Rs, rs, calc_deriv_cost)

            # (7) Accumulate total costs for all players.
            # This is the total cost for the trajectory we are on now
            #total_costs = [sum(costis).item() for costis in costs]
            print("\rTotal cost for all players:\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(total_costs[0], total_costs[1], total_costs[2]), end="")
            self._total_costs = total_costs
            
            # if max(total_costs[:2]) < 0.5 or iteration > 300:
            #     print("DONE, Enter to continue")
            #     input()
            
            #Store total cost at each iteration and the iterations
            store_total_cost.append(total_costs[0])
            iteration_store.append(iteration)
            #print("store_total_cost is: ", store_total_cost)
            
            #Plot total cost for all iterations
            if self._total_costs[0] < 0:
                plt.plot(iteration_store, store_total_cost, color='green', linestyle='dashed', linewidth = 2,
                         marker='o', markerfacecolor='blue', markersize=6)
                #plt.plot(iteration_store, store_total_cost)
                plt.xlabel('Iteration')
                plt.ylabel('Total cost')
                plt.title('Total Cost of Trajectory at Each Iteration')
                plt.show()

            # Log everything.
            if self._logger is not None:
                self._logger.log("xs", xs)
                self._logger.log("us", us)
                self._logger.log("total_costs", total_costs)
                self._logger.dump()

            # Update the member variables.
            self._Ps = Ps
            self._alphas = alphas
            self._ns = ns

            # (5) Linesearch.
            # if iteration < 5:
            #     self._alpha_scaling = 1.0
            # elif iteration < 25:
            #     self._alpha_scaling = 0.5
            # else:
            #     self._alpha_scaling = 0.1
            
            # if max(total_costs[:2]) < 1.0:
            #     self._alpha_scaling = 0.05
            # elif max(total_costs[:2]) < 0.7:
            #     self._alpha_scaling = 0.02

            self._alpha_scaling = self._linesearch_backtracking(iteration = iteration)
            
            iteration += 1

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

        # Tolerance for comparing operating points. If all states changes
        # within this tolerance in the Euclidean norm then we've converged.
        # TOLERANCE = 1e-4
        # for ii in range(self._horizon):
        #     last_x = self._last_operating_point[0][ii]
        #     current_x = self._current_operating_point[0][ii]

        #     if np.linalg.norm(last_x - current_x) > TOLERANCE:
        #         return False
            
        #     # ME TRYING SOMETHING. DELETE IF IT DOESN'T WORK!!!!!!!!!!!!
        if 1 == 1:
            return False
        
        #if self._total_costs[0] > 0.0:
        #    return False

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
    
    def _TimeStar(self, xs, us, ii):
        """
        

        Parameters
        ----------
        xs : TYPE
            DESCRIPTION.
        us : TYPE
            DESCRIPTION.

        Returns
        -------
        time_star : TYPE
            DESCRIPTION.
            
        I am mainly using this for the _linesearch_new def. This gives me the
        time_star and player cost for the hallucinated trajectories by doing the
        min-max on this trajectory

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
        
        # Pre-allocate hessian and gradient matrices
        Qs = [deque() for ii in range(self._num_players)]
        ls = [deque() for ii in range(self._num_players)]
        rs = [deque() for ii in range(self._num_players)]
        Rs = [[deque() for jj in range(self._num_players)]
              for ii in range(self._num_players)]
        
        costs = []
        
        calc_deriv_cost = deque()
        func_array = deque()
        func_return_array = deque()
    
        hold_new = 0
        target_margin_func = np.zeros((self._horizon+1, 1))
        
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

        func_key_list = [""] * (self._horizon + 1)        
        if ii == 0:
            for k in range(self._horizon, -1, -1): # T to 0
                self._player_costs[ii] = PlayerCost()
                
                hold_new = ProximityToBlockCost(g_params["car1"])(xs[k])
                target_margin_func[k] = hold_new

                max_g_func = self._CheckMultipleFunctionsP1_refactored(g_params["car1"], xs, k)
                hold_prox = max_g_func(xs[k])
                
                value_function_compare = dict()
                func_key = ""

                if k == self._horizon:
                    value_function_compare = {
                        "g_x": hold_prox,
                        "l_x": hold_new
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                else:
                    tmp = {
                        "value": value_func_plus[k+1],
                        "l_x": hold_new,
                    }
                    value_function_compare = {
                        "g_x": hold_prox,
                        min(tmp, key=tmp.get): min(tmp.values())
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                    
                if func_key == "l_x":
                    c1gc = ProximityToBlockCost(g_params["car1"], "car1_goal")
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P1 = True
                elif func_key == "g_x":
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P1 = True
                else:
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 0.0)
                    calc_deriv_cost.appendleft("False")
                    self._calc_deriv_true_P1 = False
                func_array.appendleft(func_key)
                func_return_array.appendleft(c1gc)
               
                if k == self._horizon:
                    _, r, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], np.zeros((self._num_players, self._horizon, self._num_players, 1)), k, self._calc_deriv_true_P1, ii)
                else:
                    _, r, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], [uis[k] for uis in us], k, self._calc_deriv_true_P1, ii)
    
                Qs[ii].appendleft(Q)
                ls[ii].appendleft(l)
                rs[ii].appendleft(r)
    
                for jj in range(self._num_players):
                    Rs[ii][jj].appendleft(R[jj])
                        
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P1))

            total_costs = max([c.detach().numpy().flatten()[0] for c in costs])

        # This is for P2:
        elif ii == 1:
            for k in range(self._horizon, -1, -1):
                self._player_costs[ii] = PlayerCost()
                hold_new = ProximityToBlockCost(g_params["car2"])(xs[k])
                target_margin_func[k] = hold_new
            
                max_g_func = self._CheckMultipleFunctionsP2_refactored(g_params["car2"], xs, k)                
                hold_prox = max_g_func(xs[k])
                
                value_function_compare = dict()
                func_key = ""

                if k == self._horizon:
                    value_function_compare = {
                        "g_x": hold_prox,
                        "l_x": hold_new
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                else:
                    tmp = {
                        "value": value_func_plus[k+1],
                        "l_x": hold_new,
                    }
                    value_function_compare = {
                        "g_x": hold_prox,
                        min(tmp, key=tmp.get): min(tmp.values())
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                    
                if func_key == "l_x":
                    c1gc = ProximityToBlockCost(g_params["car2"], "car2_goal")
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P2 = True
                elif func_key == "g_x":
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P2 = True
                else:
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 0.0)
                    calc_deriv_cost.appendleft("False")
                    self._calc_deriv_true_P2 = False
                func_array.appendleft(func_key)
                func_return_array.appendleft(c1gc)
                    
                if k == self._horizon:
                    _, r, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], np.zeros((self._num_players, self._horizon, self._num_players, 1)), k, self._calc_deriv_true_P2, ii)
                else:
                    _, r, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], [uis[k] for uis in us], k, self._calc_deriv_true_P2, ii)
    
                Qs[ii].appendleft(Q)
                ls[ii].appendleft(l)
                rs[ii].appendleft(r)
                
    
                for jj in range(self._num_players):
                    Rs[ii][jj].appendleft(R[jj])
                        
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P2))
                
            total_costs = max([c.detach().numpy().flatten()[0] for c in costs])
        
        elif ii == 2:
            for k in range(self._horizon, -1, -1):
                self._player_costs[ii] = PlayerCost()
                # hold_new = ProximityCost((10, 11), (10, 30), 1, name="ped_goal")(xs[k])
                hold_new = PedestrianProximityToBlockCost(g_params["ped1"], name="ped_goal")(xs[k])
                target_margin_func[k] = hold_new
            
                max_g_func = self._CheckMultipleFunctionsP3_refactored(g_params, xs, k)
                hold_prox = max_g_func(xs[k])
                value_function_compare = dict()
                func_key = ""

                if k == self._horizon:
                    value_function_compare = {
                        "g_x": hold_prox,
                        "l_x": hold_new
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                else:
                    tmp = {
                        "value": value_func_plus[k+1],
                        "l_x": hold_new,
                    }
                    value_function_compare = {
                        "g_x": hold_prox,
                        min(tmp, key=tmp.get): min(tmp.values())
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)

                if func_key == "l_x":
                    c1gc = PedestrianProximityToBlockCost(g_params["ped1"], name="ped_goal")
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P3 = True
                elif func_key == "g_x":
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P3 = True
                else:
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 0.0)
                    calc_deriv_cost.appendleft("False")
                    self._calc_deriv_true_P3 = False
                func_array.appendleft(func_key)
                func_return_array.appendleft(c1gc)
                    
                if k == self._horizon:
                    _, r, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], np.zeros((self._num_players, self._horizon, self._num_players, 1)), k, self._calc_deriv_true_P3, ii)
                else:
                    _, r, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], [uis[k] for uis in us], k, self._calc_deriv_true_P3, ii)
    
                Qs[ii].appendleft(Q)
                ls[ii].appendleft(l)
                rs[ii].appendleft(r)
    
                for jj in range(self._num_players):
                    Rs[ii][jj].appendleft(R[jj])
                        
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P3))
                
            total_costs = max([c.detach().numpy().flatten()[0] for c in costs])
        
        if "l_x" in func_key_list:
            first_lx_index = func_key_list.index("l_x")
        else:
            first_lx_index = np.inf

        if "g_x" in func_key_list:
            first_gx_index = func_key_list.index("g_x")
        else:
            first_gx_index = np.inf

        first_t_star = min(first_lx_index, first_gx_index)
        # overwrite total_costs to return the cost of the first instance of t_star
        total_costs = costs[self._horizon - first_t_star].detach().numpy().flatten()[0]
        return Qs, ls, Rs, rs, costs, total_costs, calc_deriv_cost, func_array, func_return_array
    
    
    def _CheckMultipleFunctionsP1_refactored(self, g_params, xs, k):
        max_func = dict()
        
        max_val, func_of_max_val = CollisionPenalty(g_params)(xs[k])
        max_func[func_of_max_val] = max_val
        
        max_val, func_of_max_val = RoadRulesPenalty(g_params)(xs[k])
        max_func[func_of_max_val] = max_val

        # max_val, func_of_max_val = ManeuverPenalty(g_params)(xs[k])
        return max(max_func, key=max_func.get)

    def _CheckMultipleFunctionsP2_refactored(self, g_params, xs, k):
        max_func = dict()
        
        max_val, func_of_max_val = CollisionPenalty(g_params)(xs[k])
        max_func[func_of_max_val] = max_val
        
        max_val, func_of_max_val = RoadRulesPenalty(g_params)(xs[k])
        max_func[func_of_max_val] = max_val

        # max_val, func_of_max_val = ManeuverPenalty(g_params)(xs[k])
        # max_func[func_of_max_val] = max_val

        return max(max_func, key=max_func.get)

    def _CheckMultipleFunctionsP3_refactored(self, g_params, xs, k):
        max_func = dict()
        
        max_val, func_of_max_val = PedestrianToCarCollisionPenalty(g_params["car1"])(xs[k])
        max_func[func_of_max_val] = max_val

        max_val, func_of_max_val = PedestrianToCarCollisionPenalty(g_params["car2"])(xs[k])
        max_func[func_of_max_val] = max_val
        
        return max(max_func, key=max_func.get)

    def _TimeStarRollout(self, xs, us, ii):
        """
        

        Parameters
        ----------
        xs : TYPE
            DESCRIPTION.
        us : TYPE
            DESCRIPTION.

        Returns
        -------
        time_star : TYPE
            DESCRIPTION.
            
        I am mainly using this for the _linesearch_new def. This gives me the
        time_star and player cost for the hallucinated trajectories by doing the
        min-max on this trajectory

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
        
        # Pre-allocate hessian and gradient matrices
        Qs = [deque() for ii in range(self._num_players)]
        ls = [deque() for ii in range(self._num_players)]
        rs = [deque() for ii in range(self._num_players)]
        Rs = [[deque() for jj in range(self._num_players)]
              for ii in range(self._num_players)]
        
        costs = []
        
        calc_deriv_cost = deque()
        func_array = deque()
        func_return_array = deque()
    
        hold_new = 0
        target_margin_func = np.zeros((self._horizon+1, 1))
        
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
        
        func_key_list = [""] * (self._horizon + 1)
        if ii == 0:
            for k in range(self._horizon, -1, -1): # T to 0
                self._player_costs[ii] = PlayerCost()
                
                hold_new = ProximityToBlockCost(g_params["car1"])(xs[k])
                target_margin_func[k] = hold_new

                max_g_func = self._CheckMultipleFunctionsP1_refactored(g_params["car1"], xs, k)
                hold_prox = max_g_func(xs[k])
                
                value_function_compare = dict()
                func_key = ""

                if k == self._horizon:
                    value_function_compare = {
                        "g_x": hold_prox,
                        "l_x": hold_new
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                else:
                    tmp = {
                        "value": value_func_plus[k+1],
                        "l_x": hold_new,
                    }
                    value_function_compare = {
                        "g_x": hold_prox,
                        min(tmp, key=tmp.get): min(tmp.values())
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                    
                if func_key == "l_x":
                    c1gc = ProximityToBlockCost(g_params["car1"], "car1_goal")
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P1 = True
                elif func_key == "g_x":
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P1 = True
                else:
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 0.0)
                    calc_deriv_cost.appendleft("False")
                    self._calc_deriv_true_P1 = False
                func_array.appendleft(func_key)
                func_return_array.appendleft(c1gc)
                                       
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P1))

            total_costs = max([c.detach().numpy().flatten()[0] for c in costs])

        # This is for P2:
        elif ii == 1:
            for k in range(self._horizon, -1, -1):
                self._player_costs[ii] = PlayerCost()
                hold_new = ProximityToBlockCost(g_params["car2"])(xs[k])
                target_margin_func[k] = hold_new
            
                max_g_func = self._CheckMultipleFunctionsP2_refactored(g_params["car2"], xs, k)                
                hold_prox = max_g_func(xs[k])
                
                value_function_compare = dict()
                func_key = ""

                if k == self._horizon:
                    value_function_compare = {
                        "g_x": hold_prox,
                        "l_x": hold_new
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                else:
                    tmp = {
                        "value": value_func_plus[k+1],
                        "l_x": hold_new,
                    }
                    value_function_compare = {
                        "g_x": hold_prox,
                        min(tmp, key=tmp.get): min(tmp.values())
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                    
                if func_key == "l_x":
                    c1gc = ProximityToBlockCost(g_params["car2"], "car2_goal")
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P2 = True
                elif func_key == "g_x":
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P2 = True
                else:
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 0.0)
                    calc_deriv_cost.appendleft("False")
                    self._calc_deriv_true_P2 = False
                func_array.appendleft(func_key)
                func_return_array.appendleft(c1gc)
                        
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P2))
                
            total_costs = max([c.detach().numpy().flatten()[0] for c in costs])
        
        elif ii == 2:
            for k in range(self._horizon, -1, -1):
                self._player_costs[ii] = PlayerCost()
                # hold_new = ProximityCost((10, 11), (10, 30), 1, name="ped_goal")(xs[k])
                hold_new = PedestrianProximityToBlockCost(g_params["ped1"], name="ped_goal")(xs[k])
                target_margin_func[k] = hold_new
            
                max_g_func = self._CheckMultipleFunctionsP3_refactored(g_params, xs, k)
                hold_prox = max_g_func(xs[k])
                value_function_compare = dict()
                func_key = ""

                if k == self._horizon:
                    value_function_compare = {
                        "g_x": hold_prox,
                        "l_x": hold_new
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                else:
                    tmp = {
                        "value": value_func_plus[k+1],
                        "l_x": hold_new,
                    }
                    value_function_compare = {
                        "g_x": hold_prox,
                        min(tmp, key=tmp.get): min(tmp.values())
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key = max(value_function_compare, key = value_function_compare.get)
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)

                if func_key == "l_x":
                    c1gc = PedestrianProximityToBlockCost(g_params["ped1"], name="ped_goal")
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P3 = True
                elif func_key == "g_x":
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 1.0)
                    calc_deriv_cost.appendleft("True")
                    self._calc_deriv_true_P3 = True
                else:
                    c1gc = max_g_func
                    self._player_costs[ii].add_cost(c1gc, "x", 0.0)
                    calc_deriv_cost.appendleft("False")
                    self._calc_deriv_true_P3 = False
                func_array.appendleft(func_key)
                func_return_array.appendleft(c1gc)
                        
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P3))
            
            total_costs = max([c.detach().numpy().flatten()[0] for c in costs])
                
        if "l_x" in func_key_list:
            first_lx_index = func_key_list.index("l_x")
        else:
            first_lx_index = np.inf

        if "g_x" in func_key_list:
            first_gx_index = func_key_list.index("g_x")
        else:
            first_gx_index = np.inf
        
        first_t_star = min(first_lx_index, first_gx_index)
        total_costs = costs[self._horizon - first_t_star].detach().numpy().flatten()[0]
        return first_t_star, total_costs

    def _linesearch(self, beta = 0.9, iteration = None):
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

    def _linesearch_backtracking(self, beta=0.9, iteration = None):
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
                t_star, total_cost_new = self._TimeStarRollout(xs, us, ii)
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