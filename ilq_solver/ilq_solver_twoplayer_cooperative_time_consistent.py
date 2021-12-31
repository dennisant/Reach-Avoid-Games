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
from cost.semiquadratic_polyline_cost_any import SemiquadraticPolylineCostAny
from solve_lq_game.solve_lq_game_reachavoid_timeconsistent import solve_lq_game

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
            xs, us = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, us)
            
            
            if iteration%store_freq == 0:
                
                xs_store = [xs_i.flatten() for xs_i in xs]
                #print(xs_store[0])
                #print(len(xs_store))
                #np.savetxt('horizontal_treact20_'+str(iteration)+'.out', np.array(xs_store), delimiter = ',')
                np.savetxt('threeplayer_intersection_'+str(iteration)+'.txt', np.array(xs_store), delimiter = ',')
            


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
            print("self._num_players is: ", self._num_players)
                

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
            total_costs = []
                        
                        
                        
            for ii in range(self._num_players):           
                Q, l, R, r, costs, total_costss, calc_deriv_cost_ = self._TimeStar(xs, us, ii)
                
                Qs.append(Q[ii])
                ls.append(l[ii])
                rs.append(r[ii])
                
                costs.append(costs[ii])
                calc_deriv_cost.append(calc_deriv_cost_)
                total_costs.append(total_costss)
                
                Rs.append(R[ii])
                
                        
                        
                    
                        
            self._rs = rs
            self._xs = xs
            self._us= us



            # (6) Compute feedback Nash equilibrium of the resulting LQ game.
            # This is getting put into compute_operating_point to solver
            # for the next trajectory
            Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs, rs, calc_deriv_cost)

            

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
        car_position_indices = [car1_position_indices, car2_position_indices]
        x1_index, y1_index = car1_position_indices
        x2_index, y2_index = car2_position_indices
        
        # Center of target for both players
        target1_position = (6.0, 30.0)
        target2_position = (16.0, 12.0)
        target_position = [target1_position, target2_position]
        
        # Radius of target for both players
        target1_radius = 1
        target2_radius = 1
        target_radius = [target1_radius, target2_radius]
        
        #
        car1_player_id = 0
        car2_player_id = 1
        
        # Desired separation for both players
        car1_desired_sep = 3
        car2_desired_sep = 3
        desired_sep = [car1_desired_sep, car2_desired_sep]
        
        # Defining the polylines for player 1 and 2
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
        num_func_P1 = 2
        num_func_P2 = 2
        num_func_P3 = 2
        
        
        # Pre-allocate hessian and gradient matrices
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
            print("START in P1 is here")
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
                check_funcs = self._CheckMultipleFunctionsP1(xs, num_func_P1, ii, k, car1_position_indices, car2_position_indices, car1_desired_sep, car1_polyline, lane_width)
                
                
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
                        print("goal function came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                        self._player_costs[0].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P1 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("Proximity distance came out. calc_deriv_cost should be true")
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
                else:
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
                        print("Lane boundary for P1 came out. calc_deriv_cost should be true")
                        c1gc = SemiquadraticPolylineCostAny(
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
            
            
            
            
            
            
            
            
        
        # This is for P2:
        elif ii == 1:
            print("START in P2 is here")
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
                check_funcs = self._CheckMultipleFunctionsP2(xs, num_func_P2, ii, k, car1_position_indices, car2_position_indices, car2_desired_sep, car2_polyline, lane_width)
                
                
                # 1. If the max between all g functions g_0 is the ProximityDistance from P1 to P2:
                if check_funcs[0] == 0:
                    # Calculate Proximity Distance at time-step k+1
                    print("Goal function for P2 came out")
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
                        print("Goal function came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car2_goal")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("Proximity distance for P2 came out. calc_deriv_cost should be true")
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
                                car2_desired_sep,
                                car2_player_id,
                                "car2_proximity")
                        self._player_costs[1].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P2 = False
                
                
                
                # 2. Else, if the obstacle margin function that comes out is LaneBoundary for P1:
                else:
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
                        print("Goal function for P2 came out. calc_deriv_cost should be true")
                        c1gc = ProximityCost(car_position_indices[ii], target_position[ii], target_radius[ii], "car1_goal")
                        self._player_costs[1].add_cost(c1gc, "x", 1.0)
                        calc_deriv_cost.appendleft("True")
                        self._calc_deriv_true_P2 = True
                    # Else, if value_func_plus == hold_prox, then our cost function is the LaneBoundary cost for P1
                    elif value_func_plus[k] == hold_prox:
                        #print("Obstacle margin func came out!")
                        print("Lane boundary for P2 came out. calc_deriv_cost should be true")
                        c1gc = SemiquadraticPolylineCostAny(
                            car2_polyline, lane_width, car2_position_indices,
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
                                  car1_position_indices],
                                car2_desired_sep,
                                car2_player_id,
                                "car2_proximity")
                        self._player_costs[1].add_cost(c1gc, "x", 0.0)
                        calc_deriv_cost.appendleft("False")
                        self._calc_deriv_true_P2 = False
                
                    
                    
                    
                    
                    
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
                
            print("calc_derive_cost is: ", calc_deriv_cost)
                
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
                
                
                
        
        
    #     #car1_polyline_cost = QuadraticPolylineCost(car1_polyline, car2_position_indices, "car1_polyline")
        
        return Qs, ls, Rs, rs, costs, total_costs, calc_deriv_cost
    
    
    
    
    
    
    def _CheckMultipleFunctionsP1(self, xs, num_func, ii, k, car1_position_indices, car2_position_indices, car1_desired_dist, car1_polyline, lane_width):
        hold_new_obs = np.zeros((num_func, 1)) 
        
        for ii in range(num_func):
            for j in range(1):
                if ii == 0:
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car1_position_indices, car2_position_indices, car1_desired_dist)
                    
                else:
                    #hold_new_obs[ii, j] = self._LaneBoundary(x, car1_position_indices, car1_oriented_upward, left_boundary_x, right_boundary_x)
                    a = SemiquadraticPolylineCostAny(car1_polyline, lane_width, car1_position_indices, "car1_polyline_boundary")
                    a = a(xs[k+1]).detach().numpy()
                    a = a[0][0]
                    print("aaaz is: ", a)
                    hold_new_obs[ii, j] = a
                    print("hold_new_obs[ii, j] is: ", hold_new_obs[ii, j])
                    #hold_new_obs[ii, j] = SemiquadraticPolylineCostAny(car1_polyline, lane_width, car1_position_indices, "car1_polyline_boundary")
                
       
        #if k == 19:
        #print("hold_new_obs in _CheckMultipleFunctionsP1 is: ", hold_new_obs)
        #print("hold_new_obs in CheckMultipleFunctionsP1 is: ", hold_new_obs)
       
        print("k in checkfuncP1 is: ", k)
        print("hold_new_obs in checkfuncP1 is: ", hold_new_obs)
        index = np.unravel_index(np.argmax(hold_new_obs, axis=None), hold_new_obs.shape)
        print("index in checkfuncP1 is: ", index)
        
        return index
    
    
    
        #_CheckMultipleFunctionsP2(xs[k], num_func_P2, ii, k, car1_position_indices, car2_position_indices, car2_desired_sep, car1_oriented_upward, car2_oriented_upward, car1_left_boundary_x, car1_right_boundary_x)
    def _CheckMultipleFunctionsP2(self, xs, num_func, ii, k, car1_position_indices, car2_position_indices, car2_desired_dist, car2_polyline, lane_width):
        hold_new_obs = np.zeros((num_func, 1)) 
        
        for ii in range(num_func):
            for j in range(1):
                if ii == 0:
                    hold_new_obs[ii,j] = self._ProximityDistance(xs[k+1], car1_position_indices, car2_position_indices, car2_desired_dist)
                    
                else:
                    #hold_new_obs[ii, j] = self._LaneBoundary(x, car1_position_indices, car1_oriented_upward, left_boundary_x, right_boundary_x)
                    a = SemiquadraticPolylineCostAny(car2_polyline, lane_width, car2_position_indices, "car2_polyline_boundary")
                    a = a(xs[k+1]).detach().numpy()
                    a = a[0][0]
                    #hold_new_obs[ii, j] = a(xs[k+1])
                    hold_new_obs[ii, j] = a
                
       
        #if k == 19:
        #print("hold_new_obs in _CheckMultipleFunctionsP1 is: ", hold_new_obs)
        #print("hold_new_obs in CheckMultipleFunctionsP1 is: ", hold_new_obs)
       
        print("k in checkfuncP1 is: ", k)
        print("hold_new_obs in checkfuncP1 is: ", hold_new_obs)
        index = np.unravel_index(np.argmax(hold_new_obs, axis=None), hold_new_obs.shape)
        print("index in checkfuncP1 is: ", index)
        
        return index