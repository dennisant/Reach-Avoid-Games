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

Author(s): 
    David Fridovich-Keil (dfk@eecs.berkeley.edu)
    Dennis Anthony (dennisra@princeton.edu)
    Duy Phuong Nguyen (duyn@princeton.edu)
"""
################################################################################
#
# Iterative LQ solver.
#
################################################################################

from time import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from collections import deque
from cost.maneuver_penalty import ManeuverPenalty
from cost.obstacle_penalty import ObstacleDistCost
from ilq_solver.base_solver import BaseSolver

from player_cost.player_cost import PlayerCost
from cost.proximity_cost import ProximityCost

class ILQSolver(BaseSolver):
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
                 config = None):
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
        super().__init__(dynamics, player_costs, x0, Ps, alphas, alpha_scaling, reference_deviation_weight, logger, visualizer, u_constraints, config)

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
        car_position_indices = (player_index*5, player_index*5 + 1)
        # Pre-allocate hessian and gradient matrices
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
        
        if player_index == 0:
            func_key_list = [""] * (self._horizon + 1)

            for k in range(self._horizon, -1, -1): # T to 0
                l_func_list.appendleft(
                    ProximityCost(self.l_params["car"], self.g_params["car"])
                )
                l_value_list[k] = l_func_list[0](xs[k])

                max_g_func = self._CheckMultipleGFunctions(self.g_params["car"], xs, k)
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

            # We now use the func_key_list that stores all the indices of l_x, g_x and value to determine
            # the pinch point index
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

    def _rollout(self, xs, us, player_index, **kwargs):
        """
        kwargs: if "first_t_star" is True, regardless of time consistency, pass back the total_cost = cost of first t_star
        """
        car_position_indices = (player_index*5, player_index*5+1)
        costs = []
            
        l_value_list = np.zeros((self._horizon+1, 1))
        g_value_list = np.zeros((self._horizon+1, 1))
        l_func_list = deque()
        g_func_list = deque()
        value_func_plus = np.zeros((self._horizon+1, 1))
        
        if player_index == 0:
            func_key_list = [""] * (self._horizon + 1)
            
            for k in range(self._horizon, -1, -1): # T to 0
                l_func_list.appendleft(
                    ProximityCost(self.l_params["car"], self.g_params["car"])
                )
                l_value_list[k] = l_func_list[0](xs[k])

                max_g_func = self._CheckMultipleGFunctions(self.g_params["car"], xs, k)
                g_func_list.appendleft(max_g_func)
                g_value_list[k] = g_func_list[0](xs[k])
                
                value_function_compare = dict()

                if k == self._horizon:
                    value_function_compare = {
                        "g_x": g_value_list[k],
                        "l_x": l_value_list[k]
                    }
                    value_func_plus[k] = max(value_function_compare.values())
                    func_key_list[k] = max(value_function_compare, key = value_function_compare.get)
                else:
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

    def _CheckMultipleGFunctions(self, g_params, xs, k):
        max_func = dict()
        max_val, func_of_max_val = ObstacleDistCost(g_params)(xs[k])
        max_func[func_of_max_val] = max_val
        max_val, func_of_max_val = ManeuverPenalty(g_params)(xs[k])
        max_func[func_of_max_val] = max_val

        return max(max_func, key=max_func.get)

    def _linesearch_residual(self, beta = 0.9, iteration=None):
        """ Linesearch using residual terms. """
        """
        beta (float) -> discounted term
        iteration (int) -> current iteration
        """        
        
        alpha_converged = False
        alpha = 1.0
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            
            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            t_star, total_costs_new = self._rollout(xs, us, 0)

            # Compute expected improvement for alpha = 1 (which is expressed by the offset term in the player's LQ value, v_0[t=0])
            expected_rate = self._ns[0][0]
            expected_improvement = expected_rate * alpha

            # TODO: rewrite this part to get correct expected_improvement
            expected_improvement = 0
            if total_costs_new <= self._total_costs[0] + expected_improvement:
                alpha_converged = True
                return alpha
            else:
                alpha = beta * alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small")
        
        self._alpha_scaling = alpha
        return alpha
    
    def _linesearch_trustregion(self, beta = 0.9, iteration = None, visualize_hallucination = False):
        """ Linesearch using trust region. """
        """
        beta (float) -> discounted term
        iteration (int) -> current iteration
        """        
        
        alpha_converged = False
        alpha = 1.0
        error = old_error = 0
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            
            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            
            # Visualize hallucinated traj
            if visualize_hallucination:
                plt.figure(1)
                self._visualizer.plot(trust_region=self.margin)
                plt.plot([x[0, 0] for x in xs], [x[1, 0] for x in xs],
                    "-r",
                    alpha = 0.4,
                    linewidth = 2,
                    zorder=10
                )
                plt.pause(0.001)
                plt.clf()

            new_t_star, total_costs_new = self._rollout(xs, us, 0, first_t_star=True)

            traj_diff = max([np.linalg.norm(np.array(x_new) - np.array(x_old)) for x_new, x_old in zip(np.array(xs)[:,:2,:], np.array(self._current_operating_point[0])[:,:2,:])])

            for i in range(self._num_players):
                old_t_star = self._first_t_stars[i]
                Q = self._Qs[i][old_t_star]
                q = self._ls[i][old_t_star]
                x_diff = [(np.array(x_new) - np.array(x_old)) for x_new, x_old in zip(np.array(xs)[old_t_star,:,:], np.array(self._current_operating_point[0])[old_t_star,:,:])]
                delta_cost_quadratic_approx = 0.5 * (np.transpose(x_diff) @ Q + 2 * np.transpose(q)) @ x_diff
            
            delta_cost_quadratic_actual = total_costs_new - self._total_costs[0]
            error = delta_cost_quadratic_approx - delta_cost_quadratic_actual

            if traj_diff < self.margin:
                if old_error == error:
                    alpha_converged = True
                else:
                    old_error = error
                    if abs(error) < 1.2 and abs(error) > 0.8:
                        self.margin = self.margin * 2.0
                        alpha = 1.0
                        alpha_converged = False
                    elif abs(error) >= 1.2:
                        self.margin = self.margin * 0.5
                        alpha = 1.0
                        alpha_converged = False
                    else:
                        alpha_converged = True
            else:
                alpha = beta * alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small")

            # print("Est cost: {:.5f}\tNew cost: {:.5f}\tAlpha: {:.5f}\tMargin: {:.5f}\tdelta cost: {:.5f}\tTraj diff: {:.5f}".format((delta_cost_quadratic_approx + self._total_costs[0]).flatten()[0], total_costs_new, alpha, margin, delta_cost_quadratic_approx.flatten()[0], traj_diff))
        
        self._alpha_scaling = alpha
        return alpha

    def _linesearch_trustregion_naive(self, beta = 0.9, iteration = None, visualize_hallucination = False):
        """ Linesearch using trust region. """
        """
        beta (float) -> discounted term
        iteration (int) -> current iteration
        """
        
        alpha_converged = False
        alpha = 1.0
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            
            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            
            # Visualize hallucinated traj
            if visualize_hallucination:
                plt.figure(1)
                self._visualizer.plot(trust_region=self.margin)
                plt.plot([x[0, 0] for x in xs], [x[1, 0] for x in xs],
                    "-r",
                    alpha = 0.4,
                    linewidth = 2,
                    zorder=10
                )
                plt.pause(0.001)
                plt.clf()

            new_t_star, total_costs_new = self._rollout(xs, us, 0, first_t_star=True)

            traj_diff = max([np.linalg.norm(np.array(x_new) - np.array(x_old)) for x_new, x_old in zip(np.array(xs)[:,:2,:], np.array(self._current_operating_point[0])[:,:2,:])])

            for i in range(self._num_players):
                old_t_star = self._first_t_stars[i]
                Q = self._Qs[i][old_t_star]
                q = self._ls[i][old_t_star]
                x_diff = [(np.array(x_new) - np.array(x_old)) for x_new, x_old in zip(np.array(xs)[old_t_star,:,:], np.array(self._current_operating_point[0])[old_t_star,:,:])]
                delta_cost_quadratic_approx = 0.5 * (np.transpose(x_diff) @ Q + 2 * np.transpose(q)) @ x_diff
            
            delta_cost_quadratic_actual = total_costs_new - self._total_costs[0]
            error = delta_cost_quadratic_approx - delta_cost_quadratic_actual

            if traj_diff < self.margin:
                if abs(error) < 1.2 and abs(error) > 0.8:
                    self.margin = self.margin * 2.0
                elif abs(error) >= 1.2:
                    self.margin = self.margin * 0.5
                alpha_converged = True
            else:
                alpha = beta * alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small")

            # print("Est cost: {:.5f}\tNew cost: {:.5f}\tAlpha: {:.5f}\tMargin: {:.5f}\tdelta cost: {:.5f}\tTraj diff: {:.5f}".format((delta_cost_quadratic_approx + self._total_costs[0]).flatten()[0], total_costs_new, alpha, margin, delta_cost_quadratic_approx.flatten()[0], traj_diff))

        self._alpha_scaling = alpha
        return alpha

    def _linesearch_armijo(self, beta=0.9, iteration = None):
        """ Linesearch using armijo condition. """
        """
        beta (float) -> discounted term
        iteration (int) -> current iteration
        """        
        
        alpha_converged = False
        alpha = 1.0
        
        while not alpha_converged:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            
            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            t_star, total_costs_new = self._rollout(xs, us, 0)

            if t_star < self._horizon:
                # Calculate p (delta_u in our case)
                delta_u = -self._Ps[0][t_star] @ (xs[t_star] - self._current_operating_point[0][t_star]) - self._alphas[0][t_star]
                grad_cost_u = self._rs[0][t_star]
                t = -0.5 * grad_cost_u @ delta_u
            else:
                t = 0.0
            
            if total_costs_new  + t * alpha <= self._total_costs[0]:
                alpha_converged = True
                return alpha
            else:
                alpha = beta * alpha
                if alpha < 1e-10:
                    raise ValueError("alpha too small")
        
        self._alpha_scaling = alpha
        return alpha