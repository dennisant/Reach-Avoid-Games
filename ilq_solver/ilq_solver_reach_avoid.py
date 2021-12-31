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

from player_cost.player_cost_reach_avoid import PlayerCost
from cost.proximity_cost import ProximityCost
from cost.obstacle_dist_cost import ObstacleDistCost
from solve_lq_game import solve_lq_game

class ILQSolver(object):
    def __init__(self,
                 dynamics,
                 player_costs,
                 x0,
                 Ps,
                 alphas,
                 alpha_scaling= 100.0, # 0.01,
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
        
        #self._rs
        #self._total_costs
        #self._xs
        #self._us
        
        # Insert logic to chose g or l
        
        if cost_info is not None:
            self._car_pos = cost_info[0]
            self._car_goal = cost_info[1]
            self._obs_center = cost_info[2]
            self._obs_radius = cost_info[3]
            
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
        store_freq = 1
        
        while not self._is_converged():
            # (1) Compute current operating point and update last one.
            xs, us = self._compute_operating_point()
            self._last_operating_point = self._current_operating_point
            self._current_operating_point = (xs, us)
            
            
            # Initialize each player's player cost to be blank at each new iteration
            # We need to figure out if we're in the target margin or obstacle margin
            self._player_costs[0] = PlayerCost()
            
            
            # Storing the states at every iteration
            if iteration%store_freq == 0:
                
                xs_store = [xs_i.flatten() for xs_i in xs]
                #print(xs_store[0])
                #print(len(xs_store))
                #np.savetxt('horizontal_treact20_'+str(iteration)+'.out', np.array(xs_store), delimiter = ',')
                np.savetxt('1player1obs_pinchpoint_radius4.5_'+str(iteration)+'.txt', np.array(xs_store), delimiter = ',')
            


            # Visualization.
            if self._visualizer is not None:
                traj = {"xs" : xs}
                for ii in range(self._num_players):
                    traj["u%ds" % (ii + 1)] = us[ii]

                self._visualizer.add_trajectory(iteration, traj)
                self._visualizer.plot_controls(1)
                plt.pause(0.01)
                plt.clf()
#                self._visualizer.plot_controls(2)
#                plt.pause(0.01)
#                plt.clf()
                self._visualizer.plot()
                plt.pause(0.01)
                plt.clf()
                #plt.savefig('reach-avod plots.jpg'.format(iteration)) # Trying to save these plots

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
                    
            
            # (3) Here, I get t* and which function comes out of the min-max (target function or obstacle function)
            # for the given trajectory (xs) and controls for that trajectory (us)
            time_star = self._TimeStar(xs, us)
            print("time_star is from def: ", time_star)
            
            
            # (4) This is to calculate cost of the current trajectory (now that we know if L or g came out of min-max)
            # Here I want to add in time_star since cost is zero everywhere else besides that time
            costs = [[] for ii in range(self._num_players)]
            for k in range(self._horizon):
                for ii in range(self._num_players):
                    costs[ii].append(self._player_costs[ii](
                        torch.as_tensor(xs[k].copy()),
                        [torch.as_tensor(ui) for ui in us],
                        k, time_star))
                    
                    
                    

            # (5) Quadraticize costs.
            # Get the hessians and gradients. Hess_x (Q) and grad_x (l) are zero besides at t*
            # Hess_u (R) and grad_u (r) are eps * I and eps * u_t, respectfully, for all [0, T]
            Qs = [[] for ii in range(self._num_players)]
            ls = [[] for ii in range(self._num_players)]
            rs = [[] for ii in range(self._num_players)]
            # rs = [[[] for jj in range(self._num_players)]
            #       for ii in range(self._num_players)]
            Rs = [[[] for jj in range(self._num_players)]
                  for ii in range(self._num_players)]
            for ii in range(self._num_players):
                for k in range(self._horizon):
                    _, r, l, Q, R = self._player_costs[ii].quadraticize(
                        xs[k], [uis[k] for uis in us], k, time_star)

                    Qs[ii].append(Q)
                    ls[ii].append(l)
                    rs[ii].append(r)
                    

                    for jj in range(self._num_players):
                        Rs[ii][jj].append(R[jj])
                        
            self._rs = rs
            self._xs = xs
            self._us= us
            self._time_star = time_star
                        
                        
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
            Ps, alphas = solve_lq_game(As, Bs, Qs, ls, Rs, rs)
            #print("alphas in run part of algorithm is: ", alphas)
            

            # (7) Accumulate total costs for all players.
            # This is the total cost for the trajectory we are on now
            total_costs = [sum(costis).item() for costis in costs]
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
            self._linesearch_new()
            #print("alpha is: ", self._linesearch_new())
            self._alpha_scaling = self._linesearch_new()
            # self._alphas = self._alpha_line_search
            iteration += 1
            
    def _linesearch(self, iteration, t=0.25, beta = 0.5):
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
            
            
            # if iteration < 22:
            #     alpha = 0.01
            #     return alpha
            # else:
            #     iteration += 1
            
            #1.
            alpha_converged = False
            alpha = 0.5
            
            while alpha_converged == False:
                unew = [[] for ii in range(self._num_players)]
                xs_new = [self._x0]
                #print("alpha is: ", alpha)
                #alpha = 1
                #print(self._us[0][3].shape)
                #print(self._rs[0][3].T.shape)
                
                #Need to get grad_u f:
                
                
                for k in range(self._horizon):
                    # Get new control u + alpha * p at time-step k
                    u_line_search = [self._us[0][k] - alpha * self._rs[0][k].T]
                    unew[0].append(u_line_search)
                    
                    if k == self._horizon  - 1:
                        break
                    
                    # Get next state
                    #print("xs_new[k] is: ", xs_new[k])
                    xs_new.append(self._dynamics.integrate(xs_new[k], u_line_search))
                    
                grad = 0
                for k in range(self._horizon):
                    #print("self._rs[k] is: ", self._rs[k])
                    #print("self._rs[0][k] @ self._rs[0][k].T is: ", self._rs[0][k] @ self._rs[0][k].T)
                    grad += -self._rs[0][k] @ self._rs[0][k].T
                    
                # Get cost
                costs = [[] for ii in range(self._num_players)]
                for k in range(self._horizon):
                    for ii in range(self._num_players):
                        costs[ii].append(self._player_costs[ii](
                            torch.as_tensor(xs_new[k].copy()),
                            [torch.as_tensor(ui) for ui in unew],
                            k, self._time_star))
                        
                total_costs_new = [sum(costis).item() for costis in costs]
                #print("self._total_costs is: ", self._total_costs)
                #print("total_costs_new is: ", total_costs_new)
                #print("self._total_costs + alpha * t * grad is: ", self._total_costs + alpha * t * grad)
                
                if total_costs_new < self._total_costs + alpha * t * grad:
                    alpha_converged = True
                    return alpha
                else:
                    alpha_converged = False
                    alpha = beta * alpha
            
            #self._alpha_scaling = alpha
            print("alpha is: ", alpha)
            return alpha
            #pass



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

            if k == self._horizon - 1:
                break

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
        #     if 1 == 2:
        #         return False
        
        if self._total_costs[0] > 0.0:
            return False

        return True
    
    def _ObstacleDistance(self, x, position_indices, obstacle_position, obstacle_radius):
        #x_index, y_index = position_indices
        self._x_index, self._y_index = position_indices
        self._x_position_obs, self._y_position_obs = obstacle_position
        self._obstacle_radius = obstacle_radius
        
        dx = x[self._x_index, 0] - self._x_position_obs
        dy = x[self._y_index, 0] - self._y_position_obs
    
        relative_distance = m.sqrt(dx*dx + dy*dy)
        
        return self._obstacle_radius - relative_distance #This is for reachability (eqn 7 in David's paper). Delete this one and uncomment the one below
        #return -(relative_distance - self._obstacle_radius) # This is for the reach-avoid
    
    def _TargetDistance(self, x, position_indices, target_position, target_radius):
        x_index, y_index = position_indices
        dx = x[x_index, 0] - target_position[0]
        dy = x[y_index, 0] - target_position[1]
    
        relative_distance = m.sqrt(dx*dx + dy*dy)
    
        return relative_distance - target_radius
    
    
    
    def _TimeStar(self, xs, us):
        # THIS IS ME TRYING TO FIND t* FOR REACHABILITY:
        car1_position_indices = (0,1)
        x_index, y_index = car1_position_indices
        target_position = (6.5, 35.0)
        target_radius = 2
        
        # Defining things for obstacle(s)
        obstacle_position = (6.5, 15.0)
        obstacle_radius = 6.5  # Change back to 4
    
        #1.
        # Pre-allocation for target stuff
        hold = 0
        hold_new = 0
        target_margin_func = np.zeros((self._horizon, 1))
        k_tracker = 0
        eps = 0.01
        
        # Pre-allocation for obstacle stuff
        obs_margin_func = np.zeros((self._horizon, 1))
        payoff = np.zeros((self._horizon, 1))
        t_max_obs = np.zeros((self._horizon, 1))
        
        #1a. This is for the target (checking the target distance throughout entire trajectory)
        # Once the entire trajectory is checked
        for k in range(self._horizon):
            hold_new = self._TargetDistance(xs[k], car1_position_indices, target_position, target_radius)
            target_margin_func[k] = hold_new
        
        
        
            #1b. This is for the obstacle (checking the obstacle distance from t* to T) (CHECK THIS AGAIN!!!!)
            hold_obs = -np.inf #Maybe change
            hold_new_obs = 0
            k_track_obs = 0
            for j in range(k+1): # Run look to find closest distance to obstacle from time [0, k]
                hold_new_obs = self._ObstacleDistance(xs[j], car1_position_indices, obstacle_position, obstacle_radius)
        
                if j == 0:
                    hold_obs = hold_new_obs
                    k_track_obs = j
                elif hold_new_obs > hold_obs:
                    hold_obs = hold_new_obs
                    k_track_obs = j
                    
            # 1. Store the max of g from [0, k]
            # 2. Store the time between [0, k] where g is max for each iteration
            obs_margin_func[k] = hold_obs
            t_max_obs[k] = k_track_obs
        
                    
                    
            #1c. This is me checking the max between target and obstacle margin function (Equation 4 in Jaime's Reach-Avoid 2015 paper)
            if hold_new > hold_obs:
                payoff[k] = hold_new
                #target_margin_function = True
                #k_track = k_tracker # This tells me that t* comes from target margin function
            else:
                payoff[k] = hold_obs
                #target_margin_function = False
                #k_track = k_track_obs # This tells me that t* comes from the obstacle margin function
                
        # Now, we find t when the payoff is min
        t_star = np.argmin(payoff)
        #print("payoff is: ", payoff)
        #print("t_star is: ", t_star)
        #print("obs_margin_func is: ", obs_margin_func)
        
    
        
        # Now that we have the min payoff, we need to figure out if l or g is the max at that time
        if target_margin_func[t_star] > obs_margin_func[t_star]:
            #target_margin_function = True
            #c1gc = ProximityCost(self._car_pos, self._car_goal, np.inf, "car1_goal")
            c1gc = ProximityCost(self._car_pos, self._car_goal, target_radius, "car1_goal")
            self._player_costs[0].add_cost(c1gc, "x", 1.0) #20.0 # 50 #10
            time_star = t_star
            print("we are in target_marg")
            print("target_marg_func at tau* is: ", target_margin_func[t_star])
        else:
            #target_margin_function = False
            # c1gc = ObstacleDistCost(
            #         self._car_pos, point=p, max_distance=r,
            #             name="obstacle_%f_%f" % (p.x, p.y)
            #   for p, r in zip(self._obs_center, self._obs_radius))
            #c1gc = ObstacleDistCost(
            #        self._car_pos, point=self._obs_center, max_distance=self._obs_radius, name="obstacle")
            c1gc = ObstacleDistCost(
                    self._car_pos, self._obs_center[0], obstacle_radius, name="obstacle")
            self._player_costs[0].add_cost(c1gc, "x", 1.0) #20.0 # -50.0 # -20
            time_star = int(t_max_obs[t_star])
            print("obs_marg_func at tau* is: ", obs_margin_func[time_star])
            print("we are in obstacle_marg")
            
        #print("Target margin function is: ", target_margin_function)
        #print("t* for the target function is: ", k_tracker)
        #print("The time step we're looking at is: ", k_track)
        #print("Target margin fnc is: ", target_margin_func[k_tracker])
        #print("Obstacle margin fnc is: ", obs_margin_func[k_track_obs])
        print("time_star is: ", time_star)
        #print("obs_margin_func is: ", obs_margin_func)
        #print("state is: ", xs)
        
        return time_star
    
    
    
    
    
    
    
    
    
    
    def _T_Star(self, xs, us):
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
        car1_position_indices = (0,1)
        x_index, y_index = car1_position_indices
        target_position = (6.5, 35.0)
        target_radius = 2
        
        # Defining things for obstacle(s)
        obstacle_position = (6.5, 15.0)
        obstacle_radius = 6.5  # Change back to 4
    
    
        # Pre-allocation for target stuff
        target_distance_k = 0
        target_margin_func = np.zeros((self._horizon, 1))
        
        # Pre-allocation for obstacle stuff
        obs_margin_func = np.zeros((self._horizon, 1))
        payoff = np.zeros((self._horizon, 1))
        t_max_obs = np.zeros((self._horizon, 1))
        
        
        # Make player cost empty. Since we are hallucinating trajectories, the new
        # trajectory we get, we need to now if L or g comes out of it
        self._player_costs[0] = PlayerCost()
        
        #1a. This is for the target (checking the target distance throughout entire trajectory)
        # Once the entire trajectory is checked
        for k in range(self._horizon):
            # Calculate the distance of state at time-step k from center of target
            # and then store it
            target_distance_k = self._TargetDistance(xs[k], car1_position_indices, target_position, target_radius)
            target_margin_func[k] = target_distance_k
        
        
        
            #1b. This is for the obstacle (checking the obstacle distance from 0 to k) (CHECK THIS AGAIN!!!!)
            hold_obs = -np.inf # Maybe change back to 0
            hold_new_obs = 0
            k_track_obs = 0
            for j in range(k+1): # Run look to find closest distance to obstacle from time [0, k]
                # for obs:
                #   hold_new_obs0[obs] = self._ObstacleDistance(...)   #psuedo-code
                # hold_new_obs = min(hold_new_obs0)
                # hold_new_obs_index = np.argmin(hold_new_obs0)
                hold_new_obs = self._ObstacleDistance(xs[j], car1_position_indices, obstacle_position, obstacle_radius)
        
                # If we're just starting out (time-step 0), the first one is the greatest so far
                if j == 0:
                    hold_obs = hold_new_obs
                    k_track_obs = j
                    
                # Else, if one of the next calculated obstacle distances is greater, 
                # we replace it (in hold_obs) and store time at which this happens (k_track_obs)
                elif hold_new_obs > hold_obs:
                    hold_obs = hold_new_obs
                    k_track_obs = j
                    
            # 1. Store the max of g from [0, k] and 
            # 2. Store the time between [0, k] where g is max
            obs_margin_func[k] = hold_obs
            t_max_obs[k] = k_track_obs
        
                    
                    
            #1c. This is me checking the max between target and obstacle margin function (Equation 4 in Jaime's Reach-Avoid 2015 paper)
            if target_distance_k > hold_obs:
                payoff[k] = target_distance_k
                
            else:
                payoff[k] = hold_obs
                
        # Now, we find t when the payoff is min
        t_star = np.argmin(payoff)
        
    
        
        # Now that we have the min payoff, we need to figure out if l or g is the max at that time
        if target_margin_func[t_star] > obs_margin_func[t_star]:
            c1gc = ProximityCost(self._car_pos, self._car_goal, target_radius, "car1_goal")
            self._player_costs[0].add_cost(c1gc, "x", 1.0) #20.0 # 50 #10
            time_star = t_star
            #print("we are in target_marg")
            #print("target_marg_func at tau* is: ", target_margin_func[t_star])
        
        else:
            c1gc = ObstacleDistCost(
                    self._car_pos, self._obs_center[0], self._obs_radius[0], name="obstacle")
            self._player_costs[0].add_cost(c1gc, "x", 1.0) # 20.0 # -50.0 # -20
            time_star = int(t_max_obs[t_star])
            #print("obs_marg_func at tau* is: ", obs_margin_func[time_star])
            #print("we are in obstacle_marg")
            
            
        # Print what the time_star outcome is:
        #print("time_star is: ", time_star)
        
        return time_star
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _linesearch_new(self, t=0.25, beta = 0.9):
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
        
        
        # if iteration < 30:
        #     alpha = 0.01
        #     self._alpha_scaling = alpha
        #     #iteration += 1
        #     return alpha
        #else:
            #iteration += 1
        
        
        alpha_converged = False
        alpha = 1.0
        
        while alpha_converged == False:
            # Use this alpha in compute_operating_point
            self._alpha_scaling = alpha
            #print("In line_search_new, self._alpha_scaling is: ", self._alpha_scaling)
            
            # With this alpha, compute trajectory and controls from self._compute_operating_point
            # For this trajectory, calculate t* and if L or g comes out of min-max
            xs, us = self._compute_operating_point() # Get hallucinated trajectory and controls from here
            t_star = self._T_Star(xs, us)
            
            # Calculate cost for this trajectory
            costs = [[] for ii in range(self._num_players)]
            for k in range(self._horizon):
                for ii in range(self._num_players):
                    costs[ii].append(self._player_costs[ii](
                        torch.as_tensor(xs[k].copy()),
                        [torch.as_tensor(ui) for ui in us],
                        k, t_star)) #maybe change back to self._time_star
                    
            
            # Quadraticize the cost (since we need gradients w.r.t. control u)
            # Qs = [[] for ii in range(self._num_players)]
            # ls = [[] for ii in range(self._num_players)]
            # rs = [[] for ii in range(self._num_players)]
            # # rs = [[[] for jj in range(self._num_players)]
            # #       for ii in range(self._num_players)]
            # Rs = [[[] for jj in range(self._num_players)]
            #       for ii in range(self._num_players)]
            # for ii in range(self._num_players):
            #     for k in range(self._horizon):
            #         _, r, l, Q, R = self._player_costs[ii].quadraticize(
            #             xs[k], [uis[k] for uis in us], k, t_star)

            #         Qs[ii].append(Q)
            #         ls[ii].append(l)
            #         rs[ii].append(r)
                    

            #         for jj in range(self._num_players):
            #             Rs[ii][jj].append(R[jj])
                        
            # Calculate p (delta_u in our case)
            delta_u = - self._Ps[0][self._time_star] @ ( xs[self._time_star] - self._current_operating_point[0][self._time_star] ) - self._alphas[0][self._time_star]
            
            # Determine grad cost w.r.t. u
            grad_cost_u = self._rs[0][self._time_star]
            #grad_cost_u = rs[0][t_star]

            t = -0.5 * grad_cost_u @ delta_u
            
                    
                        
            # Calculate total cost of whole trajectory (in this case, the cost is really only at t*)
            total_costs_new = [sum(costis).item() for costis in costs]
            #print("self._total_costs is: ", self._total_costs)
            #print("total_costs_new is: ", total_costs_new)
            #print("self._total_costs + alpha * t * grad is: ", self._total_costs + alpha * t * grad)
            
            # If total cost of this trajectory is less than our current trajectory,
            # then use this alpha. Else, cut alpha down by beta and repeat the above
            if total_costs_new + alpha * t <= self._total_costs:
                alpha_converged = True
                print("current trajectory cost is: ", self._total_costs)
                print("new trajectory cost is: ", total_costs_new)
                print("step-size for this is: ", alpha)
                #print("total_costs_new in _linesearch_new is: ", total_costs_new)
                #print("self._total_costs in _linesearch_new is: ", self._total_costs)
                #print("alpha scaling in _linesearch_new is: ", alpha)
                return alpha
            else:
                alpha_converged = False
                alpha = beta * alpha
                #print("total_cost_new is: ", total_costs_new)
                #print("self._total_costs is: ", self._total_costs)
                
                if alpha < 0.0000000000000001:
                    raise ValueError("alpha too small")
            
        
        self._alpha_scaling = alpha
        return alpha
        #pass