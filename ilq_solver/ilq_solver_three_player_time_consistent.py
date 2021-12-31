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
from scipy.linalg import block_diag
from collections import deque

from player_cost.player_cost_reachavoid_timeconsistent import PlayerCost
from cost.proximity_cost_reach_avoid_twoplayer import ProximityCost
from cost.distance_twoplayer_cost import ProductStateProximityCost
from resource.point import Point
from resource.polyline import Polyline
from cost.semiquadratic_polyline_cost import SemiquadraticPolylineCost
from cost.semiquadratic_polyline_cost_any import SemiquadraticPolylineCostAny
from solve_lq_game.solve_lq_game_reachavoid_timeconsistent import solve_lq_game
from utils.visualizer import Visualizer
from utils.logger import Logger

class ILQSolver(object):
    def __init__(self,
                 dynamics,
                 dynamics_10D,
                 player_costs,
                 x0,
                 Ps,
                 alphas,
                 alpha_scaling= 1.0, # 0.01,
                 reference_deviation_weight=None,
                 logger=None,
                 visualizer=None,
                 u_constraints=None,
                 cost_info=None):
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
        self._alphas = alphas
        self._u_constraints = u_constraints
        self._horizon = len(Ps[0])
        self._num_players = len(player_costs)
        
        self._dynamics_10d = dynamics_10D
        
        #self._rs
        #self._total_costs
        #self._xs
        #self._us
        
        # Insert logic to chose g or l
        
        # if cost_info is not None:
        #     self._car1_pos_indices = cost_info[0][0]
        #     self._car2_pos_indices = cost_info[0][1]
        #     self._car1_goal = cost_info[1][0]
        #     self._car2_goal = cost_info[1][1]
        #     self._car1_goal_radius = cost_info[2][0]
        #     self._car2_goal_radius = cost_info[2][1]
        #     self._car1_desired_sep = cost_info[3][0]
        #     self._car2_desired_sep = cost_info[3][1]
        #     #self._obs_center = cost_info[2]
        #     #self._obs_radius = cost_info[3]
            
        #     c1c = PlayerCost()   #player_costs[0]
        #     c2c = PlayerCost()   #player_costs[1]
            
        #     c1gc = ProximityCost(self._pos1, self._goal1, np.inf, "car1_goal")
        #     c1c.add_cost(c1gc, "x", 10.0) # -1.0
            
        #     c2gc = ProximityCost(self._pos2, self._goal2, np.inf, "car2_goal")
        #     c2c.add_cost(c2gc, "x", 10.0) # -1.0
            
        #     player_costs = [c1c, c2c]
        
        
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
        store_freq = 10
        
        while not self._is_converged():
            # # (1) Compute current operating point and update last one.
            # # The self._compute_operating_point() is the original one. Didn't want
            # # to change it, so I create another one for the case where player 1
            # # is controlling the 2nd half of the game for both players (cooperative phase)
            # if iteration == 0:
            #     xs, us = self._compute_operating_point()
            #     self._last_operating_point = self._current_operating_point
            #     self._current_operating_point = (xs, us)
            # else:
            #     xs, us = self._compute_operating_point_other(iteration)
            #     self._last_operating_point = self._current_operating_point
            #     self._current_operating_point = (xs, us)
                
            xs, us = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, us)
            
            
            if iteration%store_freq == 0:
                
                xs_store = [xs_i.flatten() for xs_i in xs]
                #print(xs_store[0])
                #print(len(xs_store))
                #np.savetxt('horizontal_treact20_'+str(iteration)+'.out', np.array(xs_store), delimiter = ',')
                np.savetxt('threeplayer_intersection_'+str(iteration)+'.txt', np.array(xs_store), delimiter = ',')
            
            
            
            
            # Initialize each player's player cost to be blank at each new iteration
            # We need to figure out if we're in the target margin or obstacle margin
            # for ii in range(self._num_players):
            #     self._player_costs[ii] = PlayerCost()
                
            # Set t_react time
            self._t_react = 15  # Change back to 10 (for time_horizon = 2.0)
            


            # Visualization.
            if self._visualizer is not None:
                traj = {"xs" : xs}
                for ii in range(self._num_players):
                    traj["u%ds" % (ii + 1)] = us[ii]

                self._visualizer.add_trajectory(iteration, traj)
                self._visualizer.plot_controls(1)
                plt.pause(0.01)
                plt.clf()
                self._visualizer.plot_controls(2)
                plt.pause(0.01)
                plt.clf()
                self._visualizer.plot()
                plt.pause(0.01)
                plt.clf()
                #plt.savefig('reach-avod plots.jpg'.format(iteration)) # Trying to save these plots
                
            
            print("plot is shown above")
                
                
            # NOTE: HERE WE NEED TO BREAK UP control 'us' into two players 
            # for the linearization and quadratization parts. After this is done,
            # then put control back as before (first half, both players are playing.
            # Second half, only player 1 is playing. So each player has control
            # size 2-by-1 for first half, then P1 has control size 4-by-1 for second half)
            # Write the code below:
            # if iteration != 0:
            #     for ii in range(self._t_react, self._horizon):
            #         a = np.vsplit(us[0][ii], 2)
            #         us[1].append(a[1])
            #         us[0][ii] = a[0]
                

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
            #Qs = [[] for ii in range(self._num_players)]
            #ls = [[] for ii in range(self._num_players)]
            #rs = [[] for ii in range(self._num_players)]
            Qs = []
            ls = []
            rs = []
            costs = []
            #Rs = [[[] for jj in range(self._num_players)]
            #      for ii in range(self._num_players)]
            Rs = []
            calc_deriv_cost = []
            total_costs = []
            # for ii in range(self._num_players):
            #     for k in range(self._horizon):
            #         _, r, l, Q, R = self._player_costs[ii].quadraticize(
            #             xs[k], [uis[k] for uis in us], k, time_star[ii], ii)

            #         Qs[ii].append(Q)
            #         ls[ii].append(l)
            #         rs[ii].append(r)
                    

            #         for jj in range(self._num_players):
            #             Rs[ii][jj].append(R[jj])
                        
                        
                        
            for ii in range(self._num_players):           
                Q, l, R, r, costs, total_costss, calc_deriv_cost_ = self._TimeStar(xs, us, ii)
                
                Qs.append(Q[ii])
                ls.append(l[ii])
                rs.append(r[ii])
                
                costs.append(costs[ii])
                calc_deriv_cost.append(calc_deriv_cost_)
                total_costs.append(total_costss)
                
                Rs.append(R[ii])
                    

                #for jj in range(self._num_players):
                #    Rs[ii][jj].append(R[jj])
                        
            
            # #print("Qs is: ", Qs)
            # #print("ls is: ", ls)
            # #print("rs is: ", rs)                                           # UNCOMMENT THESE PRINT STATEMENTS IF NEEDED
            # #print("Rs is: ", Rs)
            # #print("calc_deriv_cost[0][0] is: ", calc_deriv_cost[0][0])
            # #print("calc_deriv_cost is: ", calc_deriv_cost)
            
            # #print("Rs[0] is: ", Rs[0])
            # #print("Rs[0][0] is: ", Rs[0][0])
            # #print("Rs[0][0][0] is: ", Rs[0][0][0])
            # #print("ls[0][0].shape is: ", ls[0][0].shape)
            # #print("rs[0][0].shape is: ", rs[0][0].shape)
            
            # # Now I'm trying to build everything I need for the 2nd phase where
            # # player 1 is controlling everything. So here we need to build 
            # # B, Q, q, l and r for this phase
            # Bs_oneplayer = [[] for ii in range(self._num_players)]
            # ls_oneplayer = [[] for ii in range(self._num_players)]
            # rs_oneplayer = [[] for ii in range(self._num_players)]
            
            # # Make all Qs and ls for P2 the negative of P1 if P1 is in obstacle mode
            # # if self._target_margin_p1 == False:
            # #     for ii in range(self._horizon):
            # #         Qs[1][ii] = -Qs[0][ii]
            # #         ls[1][ii] = -ls[0][ii]
            
            # # Combine B's for 2nd phase for P1
            # for ii in range(self._t_react, self._horizon, 1):
            #     #print(Bs[0][ii])
            #     #print(Bs[1][ii])
            #     Bs[0][ii] = np.hstack((Bs[0][ii], Bs[1][ii]))
            #     Bs[1][ii] = []
                        
            
            # # Trying to put some things on P1 and empty set on P2 for times
            # # t_react to T
            # for ii in range(self._t_react, self._horizon, 1):
            #     rs[0][ii] = np.hstack((rs[0][ii], rs[1][ii]))  # For grad of cost w.r.t. u (change back to hstack)
            #     rs[1][ii] = []
                
            #     #ls[0][ii] = np.hstack((ls[0][ii], ls[1][ii])) # For grad of cost w.r.t. x
            #     #ls[1][ii] = []
                
            #     Rs[0][0][ii] = block_diag(Rs[0][0][ii], Rs[1][0][ii])
            #     Rs[0][1][ii] = block_diag(Rs[0][1][ii], Rs[1][1][ii])
            #     Rs[1][0][ii] = []
            #     Rs[1][1][ii] = []
                
            #     #Qs[0][ii] = block_diag(Qs[0][ii], Qs[1][ii])
            #     #Qs[1][ii] = []
                
                
            # #print("Qs[1] is: ", Qs[1])
                
            # # Need to stack 'us' for compute_operating_point_other for 2nd phase. 
            # # By the time we use it, the Ps and alphas are already stadcked.
            # # This really only needs to be done at iteration 0
            # self._us_other = us
            # for ii in range(self._t_react, self._horizon):
            #     self._us_other[0][ii] = np.vstack((self._us_other[0][ii], self._us_other[1][ii]))
            #     self._us_other[1][ii] = []
                
                        
                        
                    
                        
            self._rs = rs
            self._xs = xs
            self._us= us
            #self._time_star = time_star
                        
                        
            # Printing stuff
            #print("num_players is: ", self._num_players)
            #print("Qs is: ", Qs)
            #print("Q [0][19] is: ", Qs[0][19])
            #print("Q at time_star is: ", Qs[0][time_star])
            #print("Hessian matrix Q is: ", Qs)
            #print("Gradient is: ", ls)
            #print("Gradient_u rs is: ", rs)
            #print("rs is: ", rs[0])
            #print("self._us[0][19] is: ", self._us[0][19])
            #print("self._us[0][19] is: ", self._us[0][19].T.shape)
            #print("self._rs[0][19] is: ", self._rs[0][19])


            # (6) Compute feedback Nash equilibrium of the resulting LQ game.
            # This is getting put into compute_operating_point to solver
            # for the next trajectory
            print("Qs[0] before solve_lq_game is: ", Qs[0])
            print("Qs[1] before solve_lq_game is: ", Qs[1])
            Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs, rs, calc_deriv_cost)
            #print("alphas in run part of algorithm is: ", alphas)
            #print("Ps[0][3] is: ", Ps[0][3])
            #print("shape of Ps[0][17] is: ", Ps[0][3].shape)
            #print("alphas[0][3] is: ", alphas[0][3])
            #print("shape of alphas[0][3] is: ", alphas[0][3].shape)
            #print("alphas[0][13] is: ", alphas[0][13])
            #a = np.hsplit(alphas[0][13], 2)
            #print("split matrix is: ", a)
            
            

            # (7) Accumulate total costs for all players.
            # This is the total cost for the trajectory we are on now
            #total_costs = [sum(costis).item() for costis in costs]
            print("Total cost for all players: ", total_costs)
            self._total_costs = total_costs
            
            
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

            # (5) Linesearch.
            #self._linesearch()
            #self._linesearch_new(iteration)
            #print("alpha is: ", self._linesearch_new())
            #self._alpha_scaling = self._linesearch_new(iteration)
            #print("self._alpha_scaling is: ", self._alpha_scaling)
            self._alpha_scaling = 0.1
            # self._alphas = self._alpha_line_search
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
        
        # if iteration == 23:
        #     self._alpha_scaling = 0.005
        # elif iteration == 21:
        #     self._alpha_scaling = 0.001
        # elif iteration == 22:
        #     self._alpha_scaling = 0.001
        # elif iteration == 23:                #TRYING SOMETHING. DELETE
        #     self._alpha_scaling = 0.001
        # else:
        #     self._alpha_scaling = 0.01

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
                # costs[ii].append(self._player_costs[ii](
                #     torch.as_tensor(xs[k].copy()),
                #     [torch.as_tensor(ui) for ui in u],
                #     k))

            #if k == self._horizon - 1:
            #    break

            # Use 4th order Runge-Kutta to propogate system to next time-step k+1
            xs.append(self._dynamics.integrate(xs[k], u))
            
        #print("self._aplha_scaling in compute_operating_point is: ", self._alpha_scaling)
        return xs, us
    
    
    def _compute_operating_point_other(self, iteration):
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
            if k <= self._t_react - 1:
                num_players = 2
            else:
                num_players = 1
            
            
            current_x = self._current_operating_point[0][k]
            if iteration != 0:
                current_u = [self._current_operating_point[1][ii][k]
                                  for ii in range(num_players)]
            else:
                current_u = [self._us_other[ii][k]
                                  for ii in range(num_players)]
                #print("self._us_other is: ", self._us_other)

            
            # This is Eqn. 7 in the ILQGames paper
            # This gets us the control at time-step k for the updated trajectory
            feedback = lambda x, u_ref, x_ref, P, alpha, alpha_scaling : \
                       u_ref - P @ (x - x_ref) - alpha_scaling * alpha
            u = [feedback(xs[k], current_u[ii], current_x,
                      self._Ps[ii][k], self._alphas[ii][k], self._alpha_scaling[ii])   # Adding self._alpha_scaling to this (since now we have 2 players with 2 different alpha_scaling)
                 for ii in range(num_players)]
            
            
            # DELETE THIS IF IT DOESN'T WORK
            # for ii in range(num_players):
            #     u[ii] = np.clip(u[ii], a_min = -2.0, a_max = 2.0)
            

            # Append computed control (u) for the trajectory we're calculating to "us"
            for ii in range(num_players):
                us[ii].append(u[ii])

            #if k == self._horizon - 1:
            #    break

            # Use 4th order Runge-Kutta to propogate system to next time-step k+1
            if k <= self._t_react - 1:
                xs.append(self._dynamics.integrate(xs[k], u))
            else:
                xs.append(self._dynamics_10d.integrate(xs[k], u))

        #print("self._aplha_scaling in compute_operating_point is: ", self._alpha_scaling)
        return xs, us
    
    
    def _compute_operating_point_other_other(self, iteration, alpha_scaling):
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
            if k <= self._t_react - 1:
                num_players = 2
            else:
                num_players = 1
            
            
            current_x = self._current_operating_point[0][k]
            if iteration != 0:
                current_u = [self._current_operating_point[1][ii][k]
                                  for ii in range(num_players)]
            else:
                current_u = [self._us_other[ii][k]
                                  for ii in range(num_players)]
                #print("self._us_other is: ", self._us_other)

            
            # This is Eqn. 7 in the ILQGames paper
            # This gets us the control at time-step k for the updated trajectory
            feedback = lambda x, u_ref, x_ref, P, alpha, alpha_scaling : \
                       u_ref - P @ (x - x_ref) - alpha_scaling * alpha
                       
            #for ii in range(self._num_players):
            #    u = []
            #    u.append(feedback(xs[k], current_u[ii], current_x, self._Ps[ii][k], self._alphas[ii][k], alpha_scaling[ii]))
            u = [feedback(xs[k], current_u[ii], current_x,
                      self._Ps[ii][k], self._alphas[ii][k], alpha_scaling[ii])
                 for ii in range(num_players)]
            
            # DELETE THIS IF IT DOESN'T WORK
            # for ii in range(num_players):
            #     u[ii] = np.clip(u[ii], a_min = -2.0, a_max = 2.0)
            

            # Append computed control (u) for the trajectory we're calculating to "us"
            for ii in range(num_players):
                us[ii].append(u[ii])

            #if k == self._horizon - 1:
            #    break

            # Use 4th order Runge-Kutta to propogate system to next time-step k+1
            if k <= self._t_react - 1:
                xs.append(self._dynamics.integrate(xs[k], u))
            else:
                xs.append(self._dynamics_10d.integrate(xs[k], u))

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
    
    def _ProximityDistance(self, x, car1_position_indices, car2_position_indices, desired_dist):
        #x_index, y_index = position_indices
        self._car1_x_index, self._car1_y_index = car1_position_indices
        self._car2_x_index, self._car2_y_index = car2_position_indices
        self._desired_dist = desired_dist
        
        dx = x[self._car1_x_index, 0] - x[self._car2_x_index, 0]
        dy = x[self._car1_y_index, 0] - x[self._car2_y_index, 0]
    
        relative_distance = m.sqrt(dx*dx + dy*dy)
        
        return self._desired_dist - relative_distance #This is for reachability (eqn 7 in David's paper). Delete this one and uncomment the one below
        #return -(relative_distance - self._obstacle_radius) # This is for the reach-avoid
        
        
    def _ProximityDistanceAdversarial(self, x, car1_position_indices, car2_position_indices, desired_dist):
        #x_index, y_index = position_indices
        self._car1_x_index, self._car1_y_index = car1_position_indices
        self._car2_x_index, self._car2_y_index = car2_position_indices
        self._desired_dist = desired_dist
        
        dx = x[self._car1_x_index, 0] - x[self._car2_x_index, 0]
        dy = x[self._car1_y_index, 0] - x[self._car2_y_index, 0]
    
        relative_distance = m.sqrt(dx*dx + dy*dy)
        
        return relative_distance - self._desired_dist #This is for reachability (eqn 7 in David's paper). Delete this one and uncomment the one below
        #return -(relative_distance - self._obstacle_radius) # This is for the reach-avoid
        
        
    
    def _TargetDistance(self, x, position_indices, target_position, target_radius):
        x_index, y_index = position_indices
        dx = x[x_index, 0] - target_position[0]
        dy = x[y_index, 0] - target_position[1]
    
        relative_distance = m.sqrt(dx*dx + dy*dy)
    
        return relative_distance - target_radius
    
    
    
    def _LaneBoundary(self, x, position_indices, oriented_upward, left_boundary_x, right_boundary_x):
        self._x_index, self._y_index = position_indices
        self._oriented_upward = oriented_upward
        self._left_boundary_x = left_boundary_x
        self._right_boundary_x = right_boundary_x
        
        #print("x is: ", x)
        #print("self._x_index is: ", self._x_index)
        #print("x[self._x_index, 0] is: ", x[self._x_index, 0])
        #print("self._left_boundary_x is: ", self._left_boundary_x)
        
        if oriented_upward == True:
            if abs(self._left_boundary_x - x[self._y_index, 0]) <= abs(self._right_boundary_x - x[self._y_index, 0]):
                dist = self._left_boundary_x - x[self._y_index, 0]
    
            else:
                dist = x[self._y_index, 0] - self._right_boundary_x
    
        else:
            if abs(self._left_boundary_x - x[self._y_index, 0]) <= abs(self._right_boundary_x - x[self._y_index, 0]):
                dist = x[self._y_index, 0] - self._left_boundary_x
    
            else:
                dist = self._right_boundary_x - x[self._y_index, 0]
            
        
        return dist
    
    
    def _LaneBoundaryP1Adversarial(self, x, position_indices, oriented_upward, left_boundary_x, right_boundary_x):
        self._x_index, self._y_index = position_indices
        self._oriented_upward = oriented_upward
        self._left_boundary_x = left_boundary_x
        self._right_boundary_x = right_boundary_x
        
        if oriented_upward == True:
            if abs(self._left_boundary_x - x[self._x_index, 0]) <= abs(self._right_boundary_x - x[self._x_index, 0]):
                dist = x[self._x_index, 0] - self._left_boundary_x
    
            else:
                dist = self._right_boundary_x - x[self._x_index, 0]
    
        else:
            if abs(self._left_boundary_x - x[self._x_index, 0]) <= abs(self._right_boundary_x - x[self._x_index, 0]):
                dist = self._left_boundary_x - x[self._x_index, 0]
    
            else:
                dist = x[self._x_index, 0] - self._right_boundary_x
            
        
        return dist
    
    
    
    
    
    # def _TimeStar(self, xs, us, ii):
    #     # THIS IS ME TRYING TO FIND t* FOR REACHABILITY:
    #     #Car 1 and Car 2 position indices
    #     car1_position_indices = (0,1)
    #     car2_position_indices = (5,6)
    #     car_position_indices = [car1_position_indices, car2_position_indices]
    #     x1_index, y1_index = car1_position_indices
    #     x2_index, y2_index = car2_position_indices
        
    #     # Center of target for both players
    #     target1_position = (6.5, 15.0)
    #     target2_position = (4.5, 0.0)
    #     target_position = [target1_position, target2_position]
        
    #     # Radius of target for both players
    #     target1_radius = 1
    #     target2_radius = 1
    #     target_radius = [target1_radius, target2_radius]
        
    #     #
    #     car1_player_id = 0
    #     car2_player_id = 1
        
    #     # Desired separation for both players
    #     car1_desired_sep = 3
    #     car2_desired_sep = 3
    #     desired_sep = [car1_desired_sep, car2_desired_sep]
        
    #     # Defining the polylines for player 1 and 2
    #     car1_polyline = Polyline([Point(6.5, 0.0), Point(6.5, 20.0)])
    #     car2_polyline = Polyline([Point(4.5, 15.0), Point(4.5, -1.0)])
        
    
    #     #1.
    #     # Pre-allocation for target stuff
    #     hold = 0
    #     hold_new = 0
    #     target_margin_func = np.zeros((self._horizon, 1))
        
    #     # Pre-allocation for obstacle stuff
    #     prox_margin_func = np.zeros((self._horizon, 1))
    #     payoff = np.zeros((self._horizon, 1))
    #     t_max_prox = np.zeros((self._horizon, 1))
        
    #     #1a. This is for the target (checking the target distance throughout entire trajectory)
    #     # Once the entire trajectory is checked
    #     if ii == 0:
    #         for k in range(self._horizon):
    #             hold_new = self._TargetDistance(xs[k], car_position_indices[ii], target_position[ii], target_radius[ii])
    #             target_margin_func[k] = hold_new
            
            
            
    #             #1b. This is for the obstacle for P1 (checking the obstacle distance from t* to T) (CHECK THIS AGAIN!!!!)
    #             hold_prox = -np.inf #Maybe change
    #             hold_new_prox = 0
    #             k_track_prox = 0
    #             for j in range(k+1): # Run look to find closest distance to obstacle from time [0, k]
    #                 hold_new_prox = self._ProximityDistance(xs[j], car1_position_indices, car2_position_indices, desired_sep[ii])
            
    #                 if j == 0:
    #                     hold_prox = hold_new_prox
    #                     k_track_prox = j
    #                 elif hold_new_prox > hold_prox:
    #                     hold_prox = hold_new_prox
    #                     k_track_prox = j
                        
    #             # 1. Store the max of g from [0, k]
    #             # 2. Store the time between [0, k] where g is max for each iteration
    #             prox_margin_func[k] = hold_prox
    #             t_max_prox[k] = k_track_prox
            
                        
                        
    #             #1c. This is me checking the max between target and obstacle margin function (Equation 4 in Jaime's Reach-Avoid 2015 paper)
    #             if hold_new > hold_prox:
    #                 payoff[k] = hold_new
    #                 #target_margin_function = True
    #                 #k_track = k_tracker # This tells me that t* comes from target margin function
    #             else:
    #                 payoff[k] = hold_prox
    #                 #target_margin_function = False
    #                 #k_track = k_track_obs # This tells me that t* comes from the obstacle margin function
                    
    #         # Now, we find t when the payoff is min
    #         t_star = np.argmin(payoff)
    #         #print("payoff is: ", payoff)
    #         #print("t_star is: ", t_star)
    #         #print("obs_margin_func is: ", obs_margin_func)
            
        
            
    #         # Now that we have the min payoff, we need to figure out if l or g is the max at that time
    #         if target_margin_func[t_star] > prox_margin_func[t_star]:
    #             # Calculate target cost 
    #             c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                
    #             #This is me trying to add in the polyline center cost
    #             # if ii == 0:
    #             #     c2gc = QuadraticPolylineCost(car1_polyline, car1_position_indices, "car1_polyline")
    #             # else:
    #             #     c2gc = QuadraticPolylineCost(car2_polyline, car2_position_indices, "car2_polyline")
                    
                    
                
    #             self._player_costs[ii].add_cost(c1gc, "x", 1.0) #20.0 # 50 #10
    #             #self._player_costs[ii].add_cost(c2gc, "x", 1.0)
    #             time_star = t_star
    #             self._time_star_p1 = time_star
    #             self._target_margin_p1 = True
    #             print("we are in target_marg")
    #             print("target_marg_func at tau* is: ", target_margin_func[t_star])
    #         else:
    #             #target_margin_function = False
    #             # c1gc = ObstacleDistCost(
    #             #         self._car_pos, point=p, max_distance=r,
    #             #             name="obstacle_%f_%f" % (p.x, p.y)
    #             #   for p, r in zip(self._obs_center, self._obs_radius))
    #             #c1gc = ObstacleDistCost(
    #             #        self._car_pos, point=self._obs_center, max_distance=self._obs_radius, name="obstacle")
                    
    #             c1gc = ProductStateProximityCost(
    #                         [car1_position_indices,
    #                           car2_position_indices],
    #                         car1_desired_sep,
    #                         car1_player_id,
    #                         "car1_proximity")
    #             self._player_costs[ii].add_cost(c1gc, "x", 1.0) #20.0 # -50.0 # -20
    #             time_star = int(t_max_prox[t_star])
    #             self._time_star_p1 = time_star
    #             self._target_margin_p1 = False
    #             print("obs_marg_func at tau* is: ", prox_margin_func[time_star])
    #             print("we are in obstacle_marg")
                
    #         #print("Target margin function is: ", target_margin_function)
    #         #print("t* for the target function is: ", k_tracker)
    #         #print("The time step we're looking at is: ", k_track)
    #         #print("Target margin fnc is: ", target_margin_func[k_tracker])
    #         #print("Obstacle margin fnc is: ", obs_margin_func[k_track_obs])
    #         print("time_star is: ", time_star)
    #         #print("obs_margin_func is: ", obs_margin_func)
    #         #print("state is: ", xs)
            
        
    #     # This is for P2 (if ii doesn't equal 0 and it equals 1):
    #     else:
    #         # If P1 is in target margin, then:
    #         if self._target_margin_p1 == True:
    #             hold_prox = -np.inf #Maybe change
    #             hold_new_prox = 0
    #             k_track_prox = 0
    #             for j in range(self._time_star_p1 + 1): # Run look to find closest distance to obstacle from time [0, k]
    #                 hold_new_prox = self._ProximityDistanceAdversarial(xs[j], car1_position_indices, car2_position_indices, desired_sep[ii])
            
    #                 if j == 0:
    #                     hold_prox = hold_new_prox
    #                     k_track_prox = j
    #                 elif hold_new_prox < hold_prox:   # Since this is ProximityDistanceAdversarial, I changed ">" to "<"
    #                     hold_prox = hold_new_prox
    #                     k_track_prox = j
                        
    #             time_star = k_track_prox
                
    #             c2gc = ProductStateProximityCostAdversarial(
    #                         [car1_position_indices,
    #                           car2_position_indices],
    #                         car1_desired_sep,
    #                         car1_player_id,
    #                         "car1_proximity")
                
    #             self._player_costs[1].add_cost(c2gc, "x", 1.0)
                        
                
    #         # If P1 is in obstacle margin, then: 
    #         else:
    #             hold_prox = -np.inf #Maybe change
    #             hold_new_prox = 0
    #             k_track_prox = 0
    #             for j in range(self._horizon): # Run look to find closest distance to obstacle from time [0, k]
    #                 hold_new_prox = self._ProximityDistanceAdversarial(xs[j], car1_position_indices, car2_position_indices, desired_sep[ii])
            
    #                 if j == 0:
    #                     hold_prox = hold_new_prox
    #                     k_track_prox = j
    #                 elif hold_new_prox < hold_prox:  # Since this is ProximityDistanceAdversarial, I changed ">" to "<"
    #                     hold_prox = hold_new_prox
    #                     k_track_prox = j
                        
    #             time_star = k_track_prox
                
    #             c2gc = ProductStateProximityCostAdversarial(
    #                         [car1_position_indices,
    #                           car2_position_indices],
    #                         car1_desired_sep,
    #                         car1_player_id,
    #                         "car1_proximity")
                
    #             self._player_costs[1].add_cost(c2gc, "x", 1.0)
                        
            
                
        
        
    #     #car1_polyline_cost = QuadraticPolylineCost(car1_polyline, car2_position_indices, "car1_polyline")
        
    #     return time_star
    
    
    
    
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
        # THIS IS ME TRYING TO FIND t* FOR REACHABILITY:
        #Car 1 and Car 2 position indices
        car1_position_indices = (0,1)
        car2_position_indices = (5,6)
        car3_position_indices = (10,11)
        car_position_indices = [car1_position_indices, car2_position_indices, car3_position_indices]
        x1_index, y1_index = car1_position_indices
        x2_index, y2_index = car2_position_indices
        x3_index, y3_index = car3_position_indices
        
        # Center of target for both players
        target1_position = (6.0, 30.0)
        target2_position = (16.0, 12.0)
        target3_position = (10.0, 19.0)
        target_position = [target1_position, target2_position, target3_position]
        
        # Radius of target for both players
        target1_radius = 1
        target2_radius = 1
        target3_radius = 1
        target_radius = [target1_radius, target2_radius, target3_radius]
        
        #
        car1_player_id = 0
        car2_player_id = 1
        car3_player_id = 2
        
        # Desired separation for both players
        car1_desired_sep = 3
        car2_desired_sep = 3
        car3_desired_sep = 3
        desired_sep = [car1_desired_sep, car2_desired_sep, car3_desired_sep]
        
        # Defining the polylines for player 1 and 2
        #car1_polyline = Polyline([Point(8.5, -100.0), Point(8.5, 100.0)])
        #car2_polyline = Polyline([Point(4.5, 15.0), Point(4.5, -1.0)])
        #car1_polyline = Polyline([Point(10.0, 2.0), Point(-20.0, 2.0)])
        #car2_polyline = Polyline([Point(-10.0, -2.0), Point(10.0, -2.0)])
        #car3_polyline = Polyline([Point(-10.0, -2.0), Point(10.0, -2.0)])
        
        
        # New polyline
        car1_polyline = Polyline([Point(6.0, -100.0), Point(6.0, 100.0)])
        car2_polyline = Polyline([Point(2.0, 100.0),
                          Point(2.0, 18.0),
                          Point(2.5, 15.0),
                          Point(3.0, 14.0),
                          Point(5.0, 12.5),
                          Point(8.0, 12.0),
                          Point(100.0, 12.0)])
        
        # Lane width
        lane_width = 1.0
        
        
        #Other stuff
        num_func_P1 = 3
        num_func_P2 = 3
        num_func_P3 = 2
        # car1_left_boundary_x = 6.5
        # car1_right_boundary_x = 10.5
        # car2_left_boundary_x = 6.5
        # car2_right_boundary_x = 2.5
        car1_left_boundary_y = 0.0
        car1_right_boundary_y = 4.0
        car2_left_boundary_y = 0.0
        car2_right_boundary_y = -4.0
        car1_oriented_upward = True
        car2_oriented_upward = False
        store_func_P1 = np.zeros((self._horizon, 1))
        check_func_hold_target = np.zeros((self._horizon, 1))
        
        
        # Pre-allocate hessian and gradient matrices
        # Qs = [[] for ii in range(self._num_players)]
        # ls = [[] for ii in range(self._num_players)]
        # rs = [[] for ii in range(self._num_players)]
        # Rs = [[[] for jj in range(self._num_players)]
        #       for ii in range(self._num_players)]
        Qs = [deque() for ii in range(self._num_players)]
        ls = [deque() for ii in range(self._num_players)]
        rs = [deque() for ii in range(self._num_players)]
        Rs = [[deque() for jj in range(self._num_players)]
              for ii in range(self._num_players)]
        
              
        # Pre-allocate cost
        #costs = [[] for ii in range(self._num_players)]
        costs = []
        
        # Keep track if i^*_t = i^*_t+1 is true
        calc_deriv_cost = deque()
        
        
    
        #1.
        # Pre-allocation for target stuff
        hold = 0
        hold_new = 0
        target_margin_func = np.zeros((self._horizon, 1))
        
        # Pre-allocation for obstacle stuff
        # prox_margin_func = np.zeros((self._horizon, 1))
        # payoff = np.zeros((self._horizon, 1))
        # t_max_prox = np.zeros((self._horizon, 1))
        value_func_plus = np.zeros((self._horizon, 1))
        
        
        #1a. This is for the target (checking the target distance throughout entire trajectory)
        # Once the entire trajectory is checked
        if ii == 0:
            for k in range(self._horizon-1, -1, -1):
                # Clear out PlayerCost()
                self._player_costs[ii] = PlayerCost()
                
                # Calculate proximity distance between both players
                prox_distance = self._ProximityDistance(xs[k+1], car1_position_indices, car2_position_indices, car1_desired_sep)
                if prox_distance > 0.0:
                    print("zzz P2 catches P1 at time-step: ", k)
                
                # Calculate target distance at time-step k+1 for P1 and store it
                hold_new = self._TargetDistance(xs[k+1], car_position_indices[ii], target_position[ii], target_radius[ii])
                target_margin_func[k] = hold_new
            
            
                # #1b. This is for the obstacle (checking the obstacle distance from t* to T) (CHECK THIS AGAIN!!!!)
                # hold_prox = -np.inf #Maybe change
                # hold_new_prox = 0
                # k_track_prox = 0
                print("START in P1 is here")
                
                # Here, I am going through all the functions at time-step k and picking out which one is the max and at what time-step does the max occur
                # check_funcs[0] tells us which function it is (function 1, 2 or 3), and check_funcs[1] tells us at which time-step this occurs
                check_funcs = self._CheckMultipleFunctionsP1(xs, num_func_P1, ii, k, car1_position_indices, car2_position_indices, car3_position_indices, car1_desired_sep, car1_polyline, lane_width)
                
                
                # 1. If the max between all g functions g_0 is the ProximityDistance from P1 to P2:
                if check_funcs[0] == 0:
                    # Calculate Proximity Distance at time-step k+1
                    print("ProxDist for P1 came out")
                    hold_prox = self._ProximityDistance(xs[k+1], car1_position_indices, car2_position_indices, car1_desired_sep)
                    
                    # If we are at time-step self.horizon-1, then value_func = max{l_{k+1}, g{k+1}}
                    # Else, we do the max-min stuff in order to find if l_{k+1}, g{k+1} or value_func[k+1] comes out
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                        self._player_costs[0].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car1_proximity")
                        self._player_costs[0].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car1_proximity")
                        self._player_costs[0].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P1 = False
                
                
                
                # 2. Else, if the obstacle margin function that comes out is LaneBoundary for P1:
                elif check_funcs[0] == 1:
                    # The obstacle margin function in this case is the LaneBoundary function. Calculate this at time-step k+1
                    #hold_prox = self._LaneBoundary(xs[k+1], car1_position_indices, car1_oriented_upward, car1_left_boundary_y, car1_right_boundary_y)
                    hold_prox = SemiquadraticPolylineCostAny(car1_polyline, lane_width, car1_position_indices, "car1_polyline_boundary")
                    hold_prox = hold_prox(xs[k+1]).detach().numpy()
                    hold_prox = hold_prox[0][0]
                    print("Player 1 LaneBoundaryP1 came out")
                    
                    
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    # If value_func_plus[k] == hold_new, then the cost function is the ProximityDistance for P1
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                        self._player_costs[0].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    # Else, if value_func_plus == hold_prox, then our cost function is the LaneBoundary cost for P1
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("g_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = SemiquadraticPolylineCost(
                            car1_polyline, lane_width, car1_position_indices,
                            "car1_polyline_boundary")
                        self._player_costs[0].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    # Else, set it to be some function with zero weight
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car1_proximity")
                        self._player_costs[0].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P1 = False
                
                
                
                
                
                # 3. Else, if the obstacle margin function that comes out is LaneBoundary for P2:
                else:
                    # Calculate Proximity Distance at time-step k+1
                    hold_prox = self._ProximityDistance(xs[k+1], car1_position_indices, car3_position_indices, car1_desired_sep)
                    
                    # If we are at time-step self.horizon-1, then value_func = max{l_{k+1}, g{k+1}}
                    # Else, we do the max-min stuff in order to find if l_{k+1}, g{k+1} or value_func[k+1] comes out
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                        self._player_costs[0].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car3_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car1_proximity")
                        self._player_costs[0].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car1_proximity")
                        self._player_costs[0].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P1 = False
                        
                        
                
                
                # # Calculating hessians and gradients at time-step k
                # #for ii in range(self._num_players):
                # _, r, l, Q, R = self._player_costs[ii].quadraticize(
                #     xs[k], [uis[k] for uis in us], k, self._calc_deriv_true_P1, ii)
    
                # Qs[ii].append(Q)
                # ls[ii].append(l)
                # rs[ii].append(r)
                
    
                # for jj in range(self._num_players):   # I DON'T KNOW ABOUT THIS! RE-CHECK!!!!!
                #     Rs[ii][jj].append(R[jj])
                
                
               
                
               
                # Calculating hessians and gradients w.r.t. x at time-step k+1
                # Calculate hessian and gradient w.r.t. u at time-step k
                #for ii in range(self._num_players):
                _, r, l, Q, R = self._player_costs[ii].quadraticize(
                    xs[k+1], [uis[k] for uis in us], k, self._calc_deriv_true_P1, ii)
    
                Qs[ii].appendleft(Q)
                ls[ii].appendleft(l)
                rs[ii].appendleft(r)
                print("R is: ", R)
                
                if self._calc_deriv_true_P1 == True:
                    print("Q for P1 is: ", Q)
                    print("This happens at k = ", k)
                
    
                for jj in range(self._num_players):   # I DON'T KNOW ABOUT THIS! RE-CHECK!!!!!
                    Rs[ii][jj].appendleft(R[jj])
                        
                   
                #print("Rs in ilq_solver is: ", Rs)
                #print("Rs[0] in ilq_solver is: ", Rs[0])
                #print("zzz rs in ilq_solver is: ", rs)
                
                # Calculae cost at time-step k
                #for ii in range(self._num_players):
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P1))
                
            
            #print("costs is: ", costs)
            #print("zzz ls is: ", ls)
            #print("costs[0] is: ", costs[0])
            total_costs = [sum(costis).item() for costis in costs]
            total_costs = sum(total_costs)
            print("total_costs is: ", total_costs)
            print("calc_deriv_cost for P1 is: ", calc_deriv_cost)
                
            
            
            # Adding a extra one to see what happens. DELETE IF IT DOESN'T WORK
            #calc_deriv_cost.appendleft("False") 
            
            a = np.zeros((len(xs[0]), len(xs[0]))) # Change back to 0.01
            b = np.zeros((len(xs[0]), 1))
            # a= np.identity(len(xs[0])) * 0.1 # Change back to 0.01
            # b = np.zeros((len(xs[0]), 1))
            
            Qs[ii].append(a)
            ls[ii].append(b)
            
            
            
            
            
            
            
            
        
        # This is for P2 (if ii doesn't equal 0):
        elif ii == 1:
            for k in range(self._horizon-1, -1, -1):
                # Clear out PlayerCost()
                self._player_costs[ii] = PlayerCost()
                
                # Calculate proximity distance between both players
                prox_distance = self._ProximityDistance(xs[k+1], car1_position_indices, car2_position_indices, car1_desired_sep)
                if prox_distance > 0.0:
                    print("zzz P2 catches P1 at time-step: ", k)
                
                # Calculate target distance at time-step k+1 for P1 and store it
                hold_new = self._TargetDistance(xs[k+1], car_position_indices[ii], target_position[ii], target_radius[ii])
                target_margin_func[k] = hold_new
            
                
                # Here, I am going through all the functions at time-step k and picking out which one is the max and at what time-step does the max occur
                # check_funcs[0] tells us which function it is (function 1, 2 or 3), and check_funcs[1] tells us at which time-step this occurs
                check_funcs = self._CheckMultipleFunctionsP2(xs, num_func_P3, ii, k, car1_position_indices, car2_position_indices, car3_position_indices, car2_desired_sep, car2_polyline, lane_width)
                
                
                # 1. If the max between all g functions g_0 is the ProximityDistance from P1 to P2:
                if check_funcs[0] == 0:
                    # Calculate Proximity Distance at time-step k+1
                    print("ProxDist for P1 came out")
                    hold_prox = self._ProximityDistance(xs[k+1], car2_position_indices, car1_position_indices, car2_desired_sep)
                    
                    # If we are at time-step self.horizon-1, then value_func = max{l_{k+1}, g{k+1}}
                    # Else, we do the max-min stuff in order to find if l_{k+1}, g{k+1} or value_func[k+1] comes out
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car2_goal")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car2_desired_sep,
                                car2_player_id,
                                "car2_proximity")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car2_proximity")
                        self._player_costs[1].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P2 = False
                
                
                
                # 2. Else, if the obstacle margin function that comes out is LaneBoundary for P1:
                elif check_funcs[0] == 1:
                    # The obstacle margin function in this case is the LaneBoundary function. Calculate this at time-step k+1
                    #hold_prox = self._LaneBoundary(xs[k+1], car1_position_indices, car1_oriented_upward, car1_left_boundary_y, car1_right_boundary_y)
                    hold_prox = SemiquadraticPolylineCostAny(car2_polyline, lane_width, car2_position_indices, "car2_polyline_boundary")
                    hold_prox = hold_prox(xs[k+1]).detach().numpy()
                    hold_prox = hold_prox[0][0]
                    print("LaneBoundaryP2 came out")
                    
                    
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    # If value_func_plus[k] == hold_new, then the cost function is the ProximityDistance for P1
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    # Else, if value_func_plus == hold_prox, then our cost function is the LaneBoundary cost for P1
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("g_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = SemiquadraticPolylineCostAny(
                            car1_polyline, lane_width, car1_position_indices,
                            "car2_polyline_boundary")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    # Else, set it to be some function with zero weight
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car2_position_indices,
                                  car3_position_indices],
                                car2_desired_sep,
                                car2_player_id,
                                "car2_proximity")
                        self._player_costs[1].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P2 = False
                
                
                
                
                
                # 3. Else, if the obstacle margin function that comes out is LaneBoundary for P2:
                else:
                    # Calculate Proximity Distance at time-step k+1
                    hold_prox = self._ProximityDistance(xs[k+1], car2_position_indices, car3_position_indices, car2_desired_sep)
                    
                    # If we are at time-step self.horizon-1, then value_func = max{l_{k+1}, g{k+1}}
                    # Else, we do the max-min stuff in order to find if l_{k+1}, g{k+1} or value_func[k+1] comes out
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car2_goal")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProductStateProximityCost(
                                [car2_position_indices,
                                  car3_position_indices],
                                car2_desired_sep,
                                car2_player_id,
                                "car2_proximity")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car2_proximity")
                        self._player_costs[1].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P2 = False
                        
                        
                
                
                # # Calculating hessians and gradients at time-step k+1
                # #for ii in range(self._num_players):
                # _, r, l, Q, R = self._player_costs[ii].quadraticize(
                #     xs[k], [uis[k] for uis in us], k, self._calc_deriv_true_P2, ii)
    
                # Qs[ii].append(Q)
                # ls[ii].append(l)
                # rs[ii].append(r)
                    
        
                # for jj in range(self._num_players):
                #     Rs[ii][jj].append(R[jj])
                    
                    
                    
                    
                    
                # Calculating hessians and gradients at time-step k
                # Append left since we are going backwards in time
                #for ii in range(self._num_players):
                _, r, l, Q, R = self._player_costs[ii].quadraticize(
                    xs[k+1], [uis[k] for uis in us], k, self._calc_deriv_true_P2, ii)
    
                Qs[ii].appendleft(Q)
                ls[ii].appendleft(l)
                rs[ii].appendleft(r)
                
    
                for jj in range(self._num_players):   # I DON'T KNOW ABOUT THIS! RE-CHECK!!!!!
                    Rs[ii][jj].appendleft(R[jj])
                        
                        
                # Calculae cost at time-step k
                #for ii in range(self._num_players):
                # costs[ii].append(self._player_costs[ii](
                #     torch.as_tensor(xs[k].copy()),
                #     [torch.as_tensor(ui) for ui in us],
                #     k, self._calc_deriv_true_P2))
                
                # Calculate cost for function l, g or none at time-step k?
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P2))
                
                
            # Adding a extra one to see what happens. DELETE IF IT DOESN'T WORK
            #calc_deriv_cost.appendleft("False") 
            a = np.zeros((len(xs[0]), len(xs[0]))) # Change back to 0.01
            b = np.zeros((len(xs[0]), 1))
            
            # a= np.identity(len(xs[0])) * 0.1 # Change back to 0.01
            # b = np.zeros((len(xs[0]), 1))
            
            Qs[ii].append(a)
            ls[ii].append(b)
            
            #print("costs is: ", costs)
            #print("costs[0] is: ", costs[0])
            #total_costs = [sum(costis).item() for costis in costs]
            total_costs = [costis.item() for costis in costs]
            total_costs = sum(total_costs)
            print("total_costs is: ", total_costs)
            
            
            
            
            
            
            
            
            
            
        # This is for P3 (if ii doesn't equal 0 or 1):
        else:
            for k in range(self._horizon-1, -1, -1):
                # Clear out PlayerCost()
                self._player_costs[ii] = PlayerCost()
                
                # Calculate proximity distance between both players
                prox_distance_1 = self._ProximityDistance(xs[k+1], car3_position_indices, car1_position_indices, car1_desired_sep)
                prox_distance_2 = self._ProximityDistance(xs[k+1], car3_position_indices, car2_position_indices, car1_desired_sep)
                if prox_distance_1 > 0.0:
                    print("zzz P2 catches P1 at time-step: ", k)
                elif prox_distance_2 > 0.0:
                    print("zzz P2 catches P3 at time-step: ", k)
                
                # Calculate target distance at time-step k+1 for P1 and store it
                hold_new = self._TargetDistance(xs[k+1], car_position_indices[ii], target_position[ii], target_radius[ii])
                target_margin_func[k] = hold_new
            
            
                # #1b. This is for the obstacle (checking the obstacle distance from t* to T) (CHECK THIS AGAIN!!!!)
                # hold_prox = -np.inf #Maybe change
                # hold_new_prox = 0
                # k_track_prox = 0
                print("START in P1 is here")
                
                # Here, I am going through all the functions at time-step k and picking out which one is the max and at what time-step does the max occur
                # check_funcs[0] tells us which function it is (function 1, 2 or 3), and check_funcs[1] tells us at which time-step this occurs
                check_funcs = self._CheckMultipleFunctionsP3(xs, num_func_P3, ii, k, car1_position_indices, car2_position_indices, car3_position_indices, car3_desired_sep)
                
                
                # 1. If the max between all g functions g_0 is the ProximityDistance:
                if check_funcs[0] == 0:
                    # Calculate Proximity Distance at time-step k+1
                    hold_prox = self._ProximityDistance(xs[k+1], car1_position_indices, car3_position_indices, car3_desired_sep)
                    
                    # If we are at time-step self.horizon-1, then value_func = max{l_{k+1}, g{k+1}}
                    # Else, we do the max-min stuff in order to find if l_{k+1}, g{k+1} or value_func[k+1] comes out
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car3_goal")
                        self._player_costs[2].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car3_position_indices],
                                car3_desired_sep,
                                car3_player_id,
                                "car3_proximity")
                        self._player_costs[2].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car2_position_indices],
                                car1_desired_sep,
                                car1_player_id,
                                "car3_proximity")
                        self._player_costs[2].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P1 = False
                
                
                
                # 2. Else, if the obstacle margin function that comes out is LaneBoundary for P1:
                else:
                    # Calculate Proximity Distance at time-step k+1
                    hold_prox = self._ProximityDistance(xs[k+1], car3_position_indices, car2_position_indices, car3_desired_sep)
                    
                    # If we are at time-step self.horizon-1, then value_func = max{l_{k+1}, g{k+1}}
                    # Else, we do the max-min stuff in order to find if l_{k+1}, g{k+1} or value_func[k+1] comes out
                    if k == self._horizon - 1:
                        value_func_plus[k] = np.max((hold_new, hold_prox))
                    else:
                        value_func_plus[k] = np.max( (hold_prox, np.min((hold_new, value_func_plus[k+1])) ) )
                        
                    
                    # Now figure out if l, g or V comes out of max{g_k, min{l_k, V_k^+}}
                    if value_func_plus[k] == hold_new:
                        #print("Target margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                        self._player_costs[2].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("l_{k+1} came out. calc_deriv_cost should be true")
                        c1gc = ProductStateProximityCost(
                                [car1_position_indices,
                                  car3_position_indices],
                                car3_desired_sep,
                                car3_player_id,
                                "car3_proximity")
                        self._player_costs[2].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    else:
                        #print("Value func came out!")
                        print("value_func[k] = value_func[k+1] came out. calc_deriv_cost should be False")
                        c1gc = ProductStateProximityCost(
                                [car2_position_indices,
                                  car3_position_indices],
                                car3_desired_sep,
                                car3_player_id,
                                "car3_proximity")
                        self._player_costs[2].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P1 = False
                        
                        
                
                
                # # Calculating hessians and gradients at time-step k
                # #for ii in range(self._num_players):
                # _, r, l, Q, R = self._player_costs[ii].quadraticize(
                #     xs[k], [uis[k] for uis in us], k, self._calc_deriv_true_P1, ii)
    
                # Qs[ii].append(Q)
                # ls[ii].append(l)
                # rs[ii].append(r)
                
    
                # for jj in range(self._num_players):   # I DON'T KNOW ABOUT THIS! RE-CHECK!!!!!
                #     Rs[ii][jj].append(R[jj])
                
                
               
                
               
                # Calculating hessians and gradients w.r.t. x at time-step k+1
                # Calculate hessian and gradient w.r.t. u at time-step k
                #for ii in range(self._num_players):
                _, r, l, Q, R = self._player_costs[ii].quadraticize(
                    xs[k+1], [uis[k] for uis in us], k, self._calc_deriv_true_P1, ii)
    
                Qs[ii].appendleft(Q)
                ls[ii].appendleft(l)
                rs[ii].appendleft(r)
                
                if self._calc_deriv_true_P1 == True:
                    print("Q for P1 is: ", Q)
                    print("This happens at k = ", k)
                
    
                for jj in range(self._num_players):   # I DON'T KNOW ABOUT THIS! RE-CHECK!!!!!
                    Rs[ii][jj].appendleft(R[jj])
                        
                   
                #print("Rs in ilq_solver is: ", Rs)
                #print("Rs[0] in ilq_solver is: ", Rs[0])
                #print("zzz rs in ilq_solver is: ", rs)
                
                # Calculae cost at time-step k
                #for ii in range(self._num_players):
                costs.append(self._player_costs[ii](
                    torch.as_tensor(xs[k].copy()),
                    [torch.as_tensor(ui) for ui in us],
                    k, self._calc_deriv_true_P1))
                
            
            #print("costs is: ", costs)
            #print("zzz ls is: ", ls)
            #print("costs[0] is: ", costs[0])
            total_costs = [sum(costis).item() for costis in costs]
            total_costs = sum(total_costs)
            print("total_costs is: ", total_costs)
            print("calc_deriv_cost for P1 is: ", calc_deriv_cost)
                
            
            
            # Adding a extra one to see what happens. DELETE IF IT DOESN'T WORK
            #calc_deriv_cost.appendleft("False") 
            
            a = np.zeros((len(xs[0]), len(xs[0]))) # Change back to 0.01
            b = np.zeros((len(xs[0]), 1))
            # a= np.identity(len(xs[0])) * 0.1 # Change back to 0.01
            # b = np.zeros((len(xs[0]), 1))
            
            Qs[ii].append(a)
            ls[ii].append(b)
                
                
                
        
        
    #     #car1_polyline_cost = QuadraticPolylineCost(car1_polyline, car2_position_indices, "car1_polyline")
        
        return Qs, ls, Rs, rs, costs, total_costs, calc_deriv_cost
    
    
    
    
    
    
    def _CheckMultipleFunctionsP1(self, xs, num_func, ii, k, car1_position_indices, car2_position_indices, car3_position_indices, car1_desired_dist, car1_polyline, lane_width):
        hold_new_obs = np.zeros((num_func, 1)) 
        
        for ii in range(num_func):
            for j in range(1):
                if ii == 0:
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car1_position_indices, car2_position_indices, car1_desired_dist)
                    
                elif ii == 1:
                    #hold_new_obs[ii, j] = self._LaneBoundary(x, car1_position_indices, car1_oriented_upward, left_boundary_x, right_boundary_x)
                    a = SemiquadraticPolylineCostAny(car1_polyline, lane_width, car1_position_indices, "car1_polyline_boundary")
                    a = a(xs[k+1]).detach().numpy()
                    a = a[0][0]
                    hold_new_obs[ii, j] = a
                    print("hold_new_obs[ii, j] is: ", hold_new_obs[ii, j])
                    #hold_new_obs[ii, j] = SemiquadraticPolylineCostAny(car1_polyline, lane_width, car1_position_indices, "car1_polyline_boundary")
                    
                else:
                    #hold_new_obs[ii, j] = self._LaneBoundary(x, car2_position_indices, car2_oriented_upward, left_boundary_x, right_boundary_x)
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car1_position_indices, car3_position_indices, car1_desired_dist)
                
       
        #if k == 19:
        #print("hold_new_obs in _CheckMultipleFunctionsP1 is: ", hold_new_obs)
        #print("hold_new_obs in CheckMultipleFunctionsP1 is: ", hold_new_obs)
       
        print("k in checkfuncP1 is: ", k)
        print("hold_new_obs in checkfuncP1 is: ", hold_new_obs)
        index = np.unravel_index(np.argmax(hold_new_obs, axis=None), hold_new_obs.shape)
        print("index in checkfuncP1 is: ", index)
        
        return index
    
    
    
        #_CheckMultipleFunctionsP2(xs[k], num_func_P2, ii, k, car1_position_indices, car2_position_indices, car2_desired_sep, car1_oriented_upward, car2_oriented_upward, car1_left_boundary_x, car1_right_boundary_x)
    def _CheckMultipleFunctionsP2(self, xs, num_func, ii, k, car1_position_indices, car2_position_indices, car3_position_indices, car2_desired_dist, car2_polyline, lane_width):
        hold_new_obs = np.zeros((num_func, 1)) 
        
        for ii in range(num_func):
            for j in range(1):
                if ii == 0:
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car1_position_indices, car2_position_indices, car2_desired_dist)
                    
                elif ii == 1:
                    #hold_new_obs[ii, j] = self._LaneBoundary(x, car1_position_indices, car1_oriented_upward, left_boundary_x, right_boundary_x)
                    a = SemiquadraticPolylineCostAny(car2_polyline, lane_width, car2_position_indices, "car2_polyline_boundary")
                    a = a(xs[k+1]).detach().numpy()
                    a = a[0][0]
                    hold_new_obs[ii, j] = a
                    
                else:
                    #hold_new_obs[ii, j] = self._LaneBoundary(x, car2_position_indices, car2_oriented_upward, left_boundary_x, right_boundary_x)
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car2_position_indices, car3_position_indices, car2_desired_dist)
                
       
        #if k == 19:
        #print("hold_new_obs in _CheckMultipleFunctionsP1 is: ", hold_new_obs)
        #print("hold_new_obs in CheckMultipleFunctionsP1 is: ", hold_new_obs)
       
        print("k in checkfuncP1 is: ", k)
        print("hold_new_obs in checkfuncP1 is: ", hold_new_obs)
        index = np.unravel_index(np.argmax(hold_new_obs, axis=None), hold_new_obs.shape)
        print("index in checkfuncP1 is: ", index)
        
        return index
    
    
    
    
    def _CheckMultipleFunctionsP3(self, xs, num_func, ii, k, car1_position_indices, car2_position_indices, car3_position_indices, car3_desired_dist):
        hold_new_obs = np.zeros((num_func, 1)) 
        
        for ii in range(num_func):
            for j in range(1):
                if ii == 0:
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car3_position_indices, car1_position_indices, car3_desired_dist)
                    
                else:
                    #hold_new_obs[ii, j] = self._LaneBoundary(x, car1_position_indices, car1_oriented_upward, left_boundary_x, right_boundary_x)
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car3_position_indices, car2_position_indices, car3_desired_dist)
                
       
        #if k == 19:
        #print("hold_new_obs in _CheckMultipleFunctionsP1 is: ", hold_new_obs)
        #print("hold_new_obs in CheckMultipleFunctionsP1 is: ", hold_new_obs)
       
        print("k in checkfuncP1 is: ", k)
        print("hold_new_obs in checkfuncP1 is: ", hold_new_obs)
        index = np.unravel_index(np.argmax(hold_new_obs, axis=None), hold_new_obs.shape)
        print("index in checkfuncP1 is: ", index)
        
        return index
    
    
    
    
    
    
    
    def _linesearch_new(self, iteration, t=0.25, beta = 0.9):
        """ Linesearch for both players separately. """
        """
        x -> us
        p -> rs
        may need xs to compute trajectory
        Line search needs c and tau (c = tau = 0.5 default)
        m -> local slope (calculate in function)
        need compute_operating_point routine
        """
        # p(0) = -grad f(u) = -rs
        
        # Calculate m at each iteration
        # Calculate t at each iteration
        # We can pass in f(x)
        # Need routine to calculate f(u + alpha * p)
        #   1. Get new trajectory (xs, us = compute_operating_point())
        #   2. Calculate cost of trajectory f(u + alpha * p) (pull cost from above, need t*)
        #       2a. time_star = self._TimeStar(xs, us)
        #       2b. costs = self._player_costs(xs, us, player_num, time_star)
        #   3. Check the armijo condition
        #   4. 
        #       4a. If satisfied, return alpha
        #       4b. If not satisfied, scale down alpha and repeat process
        #   5. Get new p
        #       5a. self._player_costs.quadraticize(xs, us, k, time_star)
        
        
        
        # This is me trying to do the leader-follower linesearch game
        
        # 1. Define the set of alphas we'll be going through for both players
        #alpha_car1 = [1, 0.7, 0.5, 0.3, 0.1, 0.01]
        #alpha_car2 = [1, 0.7, 0.5, 0.3, 0.1, 0.01]
        alpha_car1 = [0.2]
        alpha_car2 = [0.2]
        #alpha_car1 = [1, 0.5, 0.3, 0.1]
        #alpha_car2 = [1, 0.5, 0.3, 0.1]
        
        P1_cost = [[] for ii in range(len(alpha_car1))]
        P2_cost = [[] for ii in range(len(alpha_car2))]
        #total_costs = []
        
        
        
        for ll in range(len(alpha_car1)): # Loop through all alphas for leader (car 1)
            for mm in range(len(alpha_car1)): # Loop trhough all alphas for follower (car 2)
                # Clear out total costs
                total_costs = []
                
                # Select alphas that we will be putting in compute_operating_point
                # to get hallucinated trajectories
                alpha_scaling_set = [alpha_car1[ll], alpha_car2[mm]]
                xs, us = self._compute_operating_point_other_other(iteration, alpha_scaling_set)
                #print("us is: ", us)
                print("We are in linesearch_new")
                
                # Calculate t* for each player
                #t_star = []
                #for k in range(self._num_players):
                #    t_star.append( self._T_Star(xs, us, k) )
                    
                    
                for ii in range(self._t_react, self._horizon):
                    a = np.vsplit(us[0][ii], 2)
                    us[1].append(a[1])
                    us[0][ii] = a[0]
                    
                #print("t_star in _linesearch_new is: ", t_star)
                #print("t_star[0] is: ", t_star[0])
                #print("len(us) in _linesearch_new is: ", len(us))
                #print("us in _linesearch_new is: ", us)
                    
                # Calculate cost for trajectories
                # costs = [[] for ii in range(self._num_players)]
                # for k in range(self._horizon):
                #     #print("k is: ", k)
                #     for ii in range(self._num_players):
                #         costs[ii].append(self._player_costs[ii](
                #             torch.as_tensor(xs[k].copy()),
                #             [torch.as_tensor(ui) for ui in us],
                #             k, t_star[ii]))
                
                for kk in range(self._num_players):
                    Q, l, R, r, costs, total_costss, calc_deriv_cost_ = self._TimeStar(xs, us, kk)
                    total_costs.append(total_costss)
                        
                
                #total_costs = [sum(costis).item() for costis in costs]
                P1_cost[ll].append(total_costs[0])
                P2_cost[ll].append(total_costs[1])
                
            
        #print("zzz end of total!")
        P1_costs = np.array(P1_cost)
        P2_costs = np.array(P2_cost)
        print("P2_costs is: ", P2_costs)
        #index_array = np.argmin(P1_costs, axis=-1)
        # index_array = np.argmax(P1_costs, axis=-1)
        
        
        # #store_position_ind = [(0, index_array[0]), (1, index_array[1]), (2 ,index_array[2]), (3, index_array[3]), (4, index_array[4]), (5, index_array[5])]
        
        # store_position_ind = []
        # for ii in range(len(alpha_car1)):
        #     store_position_ind.append((ii, index_array[ii]))
        
        # mins = np.take_along_axis(P1_costs, np.expand_dims(index_array, axis=-1), axis=-1)
                
        # #maxes_of_mins = np.argmax(mins)
        # maxes_of_mins = np.argmin(mins)
        
        # hold = store_position_ind[maxes_of_mins]
        
        
        
        
        index_array = np.argmin(P2_costs, axis=-1)
        
        
        #store_position_ind = [(0, index_array[0]), (1, index_array[1]), (2 ,index_array[2]), (3, index_array[3]), (4, index_array[4]), (5, index_array[5])]
        
        store_position_ind = []
        for ii in range(len(alpha_car1)):
            store_position_ind.append((ii, index_array[ii]))
            
        print("store_position_ind is: ", store_position_ind)
        
        mins = np.take_along_axis(P2_costs, np.expand_dims(index_array, axis=-1), axis=-1)
                
        maxes_of_mins = np.argmax(mins)
        #maxes_of_mins = np.argmin(mins)
        
        hold = store_position_ind[maxes_of_mins]
        
        
        
        
        
        
        alpha_p1 = hold[0]
        alpha_p2 = hold[1]
        
        alpha_p1 = alpha_car1[alpha_p1]
        alpha_p2 = alpha_car2[alpha_p2]
        
        alpha = [alpha_p1, alpha_p2]
        print("alphas are: ", alpha)
        
        #alpha_max_p1 = np.unravel_index(np.argmax(P1_costs, axis=None), P1_costs.shape)
                        
        
        #alpha = 0.5
        
        return alpha
        #pass