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
           Chia-Yin Shih        ( cshih@berkeley.edu )
"""
################################################################################
#
# Container to hold a bunch of different Costs and keep track of the arguments
# to each one. Supports automatic quadraticization.
#
################################################################################

import numpy as np
import torch

from cost import Cost

class PlayerCost(object):
    def __init__(self):
        self._costs = []
        self._args = []
        self._weights = []

    def __call__(self, x, u, k, calc_deriv_cost):
        """
        Evaluate the game cost function at the current state and controls.
        NOTE: `x`, each `u` are all column vectors.

        :param x: state of the system
        :type x: np.array or torch.Tensor
        :param u: list of control inputs for each player
        :type u: [np.array] or [torch.Tensor]
        :param k: time step, if cost is time-varying
        :type k: uint
        :return: scalar value of cost
        :rtype: float or torch.Tensor
        """
        first_time_through = True
        for cost, arg, weight in zip(self._costs, self._args, self._weights):
            if arg == "x":
                cost_input = x
            else:
                cost_input = u[arg]
                
                
            global total_cost
            #Print the cost input (DELETE THIS WHEN DONE):
            #print("time_star in player_cost is: ", time_star)
            #print("This is for k = ", k)
            #print("self._costs is: ", self._costs)
            #print("This is for the cost: ", cost._name)
            #print("cost input is: ", cost_input)
            #print("weight is: ", weight)
            #print("The cost(cost_input,k) is: ", cost(cost_input,k))
            
            if calc_deriv_cost == True:
                current_term = weight * cost(cost_input, k)
                
            else:
                weight = 0.0
                current_term = weight * cost(cost_input, k)
                
            
            #print("current_term in player_cost is: ", current_term)
            #current_term = weight * cost(cost_input, k)
            if current_term > 1e8:
                print("Warning: cost %s is %f" % (cost._name, current_term))
                print("Input is: ", cost_input)

#            if cost._name[:4] == "bike":
#                print(cost._name, ": ", current_term)

            if first_time_through:
                total_cost = current_term
            else:
                total_cost += current_term

            first_time_through = False
            

        return total_cost

    def add_cost(self, cost, arg, weight=1.0):
        """
        Add a new cost to the game, and specify its argument to be either
        "x" or an integer indicating which player's control it is, e.g. 0
        corresponds to u0. Also assign a weight.

        :param cost: cost function to add
        :type cost: Cost
        :param arg: argument of cost, either "x" or a player index
        :type arg: string or uint
        :param weight: multiplicative weight for this cost
        :type weight: float
        """
        self._costs.append(cost)
        self._args.append(arg)
        self._weights.append(weight)
        
    def remove_cost(self, cost, arg, weight=1.0):
        """
        Add a new cost to the game, and specify its argument to be either
        "x" or an integer indicating which player's control it is, e.g. 0
        corresponds to u0. Also assign a weight.

        :param cost: cost function to add
        :type cost: Cost
        :param arg: argument of cost, either "x" or a player index
        :type arg: string or uint
        :param weight: multiplicative weight for this cost
        :type weight: float
        """
        self._costs.remove(cost)
        self._args.remove(arg)
        self._weights.remove(weight)

    def quadraticize(self, x, u, k, calc_deriv_cost, ii):
        """
        Compute a quadratic approximation to the overall cost for a
        particular choice of state `x`, and controls `u` for each player.

        Returns the gradient and Hessian of the overall cost such that:
        ```
           cost(x + dx, [ui + dui], k) \approx
                cost(x, u1, u2, k) +
                grad_x^T dx +
                0.5 * (dx^T hess_x dx + sum_i dui^T hess_ui dui)
        ```

        NOTE that in the notation of `solve_lq_game.py`, for player i:
          * `grad_x = li`
          * `hess_x = Qi`
          * `hess_uj = Rij`

        :param x: state
        :type x: np.array
        :param u: list of control inputs for each player
        :type u: np.array
        :param k: time step, if cost is time-varying
        :type k: uint
        :return: cost(x, u), grad_x, hess_x, [hess_ui]
        :rtype: float, np.array, np.array, [np.array]
        """
        # # THIS WAS THE ORIGINAL WITH POTENTIAL MULTIPLE PLAYERS
        # num_players = len(u)
        
        # #print("u[0] is: ", u[0])
        # print("u is: ", u)
        # # for ui in u:
        # #     print("ui is: ", ui)

        # # Congert to torch.Tensor format.
        # x_torch = torch.from_numpy(x).requires_grad_(True)
        # u_torch = [torch.from_numpy(ui).requires_grad_(True) for ui in u]

        # # Evaluate cost here.
        # cost_torch = self.__call__(x_torch, u_torch, k, time_star)
        # cost = cost_torch.item()

        # # Compute gradients (and store numpy versions).
        # grad_x_torch = torch.autograd.grad(
        #     cost_torch, x_torch, create_graph=True, allow_unused=True)[0]
        # grad_u_torch = [
        #     torch.autograd.grad(
        #         cost_torch, ui_torch, create_graph=True, allow_unused=True)[0]
        #     for ui_torch in u_torch]

        # # Compute Hessians (and store numpy versions), and be careful to
        # # catch Nones (which indicate cost not depending on a particular
        # # variable).
        # hess_x = np.zeros((len(x), len(x)))
        # grad_x = np.zeros((len(x), 1))
        # if grad_x_torch is not None:
        #     grad_x = grad_x_torch.detach().numpy().copy()
        #     for ii in range(len(x)):
        #         hess_row = torch.autograd.grad(
        #             grad_x_torch[ii, 0], x_torch, retain_graph=True)[0]
        #         hess_x[ii, :] = hess_row.detach().numpy().copy().T

        # hess_u = []
        # for ii in range(num_players):
        #     hess_ui = np.zeros((len(u[ii]), len(u[ii])))
        #     grad_ui_torch = grad_u_torch[ii]
        #     if grad_ui_torch is not None:
        #         grad_ui = grad_ui_torch.detach().numpy().copy()
        #         for dim in range(len(u[ii])):
        #             hess_row = torch.autograd.grad(
        #                 grad_ui_torch[dim, 0], u_torch[ii], retain_graph=True)[0]
        #             hess_ui[dim, :] = hess_row.detach().numpy().copy().T

        #     hess_ui = hess_ui + np.identity(len(u[ii])) * 0.6
        #     hess_u.append(hess_ui)
            
            
            
        # #grad_u = []
        # gradient_u = np.zeros((num_players, len(u[0])))
        # #print("len(u) is: ", len(u))
        # for ii in range(num_players):
        #     grad_ui_torch = grad_u_torch[ii]
        #     if grad_ui_torch is not None:
        #         gradient_u[ii, :] = grad_ui_torch.detach().numpy().copy().T
        #         #grad_u.append(gradient_u)
        #         #grad_ui = grad_ui_torch.detach().numpy().copy()
        #         #grad_u.append(grad_ui)
        #     #print("grad_u so far is: ", grad_u)
                
        # #print("grad_u is: ", grad_u)
            
            
            
        # #grad_u = grad_ui
        # #print("grad_u is: ", grad_u)


        # #hess_x = hess_x + np.identity(len(x)) * 0.2
        
        # # #print("eig of hess_x is: ", np.linalg.eig(hess_x))
        # # #print("eig of hess_u is: ", np.linalg.eig(hess_u[0]))
        # # #print("hess_x is: ", hess_x)

        
        
        
        
        
        
        
        
        
        # THIS IS THE REACH-AVOID FOR 1-PLAYER CASE:
        if calc_deriv_cost == True:
            num_players = len(u)

            # Congert to torch.Tensor format.
            x_torch = torch.from_numpy(x).requires_grad_(True)
            u_torch = [torch.from_numpy(ui).requires_grad_(True) for ui in u]

            # Evaluate cost here.
            cost_torch = self.__call__(x_torch, u_torch, k, calc_deriv_cost)
            cost = cost_torch.item()

            # Compute gradients (and store numpy versions).
            # First, we calculate grad_x and grad_u
            grad_x_torch = torch.autograd.grad(
                cost_torch, x_torch, create_graph=True, allow_unused=True)[0]
            grad_u_torch = [
                torch.autograd.grad(
                    cost_torch, ui_torch, create_graph=True, allow_unused=True)[0]
                for ui_torch in u_torch]

            # Compute Hessians (and store numpy versions), and be careful to
            # catch Nones (which indicate cost not depending on a particular
            # variable).
            hess_x = np.zeros((len(x), len(x)))
            grad_x = np.zeros((len(x), 1))
            if grad_x_torch is not None:
                grad_x = grad_x_torch.detach().numpy().copy()
                for ii in range(len(x)):
                    hess_row = torch.autograd.grad(
                        grad_x_torch[ii, 0], x_torch, retain_graph=True)[0]  # Added allow_unused=True. Delete that part if not true
                    hess_x[ii, :] = hess_row.detach().numpy().copy().T

            hess_u = []
            for ii in range(num_players):
                hess_ui = np.zeros((len(u[ii]), len(u[ii])))
                grad_ui_torch = grad_u_torch[ii]
                if grad_ui_torch is not None:
                    grad_ui = grad_ui_torch.detach().numpy().copy()
                    for dim in range(len(u[ii])):
                        hess_row = torch.autograd.grad(
                            grad_ui_torch[dim, 0], u_torch[ii], retain_graph=True)[0]
                        print("hess_row is: ", hess_row)
                        hess_ui[dim, :] = hess_row.detach().numpy().copy().T
                        print("hess_ui[dim, :] is: ", hess_ui[dim, :])

                hess_u.append(hess_ui)
                
                
            gradient_u = np.zeros((num_players, len(u[0])))
            #print("len(u) is: ", len(u))
            for ii in range(num_players):
                grad_ui_torch = grad_u_torch[ii]
                if grad_ui_torch is not None:
                    gradient_u[ii, :] = grad_ui_torch.detach().numpy().copy().T
                    #grad_u.append(gradient_u)
                    #grad_ui = grad_ui_torch.detach().numpy().copy()
                    #grad_u.append(grad_ui)
                #print("grad_u so far is: ", grad_u)
                    
            #print("grad_u is: ", grad_u)
             
                
             
            hess_x = hess_x + np.identity(len(x)) * 0.1 # Added 0.01 to this. Delete if doesn't work
            hess_u = [np.identity(2) * 0.1, np.identity(2) * 0.1]
            
            #gradient_u = np.zeros((2,2))
            #gradient_u[0,:] = 0.3 * u[0].T
            #gradient_u[1,:] = 0.3 * u[1].T
            
            gradient_u = np.vstack((0.1 * u[0].T, 0.1 * u[1].T))
            
            
            # if ii == 0:
            #     gradient_u = 0.2 * u[0].T
            #     hess_u = [np.identity(2) * 0.2]       #TRYING SOMETHING OUT HERE. DELETE IF INCORRECT!!!!!!!
            # else:
            #     gradient_u = 0.2 * u[1].T
            #     hess_u = [np.identity(2) * 0.2]
            
            
            # if ii == 0:
            #     gradient_u = np.vstack((0.2 * u[0].T, 0.0 * u[1].T))
            #     hess_u = [np.identity(2) * 0.2, np.zeros((2,2))]       #TRYING SOMETHING OUT HERE. DELETE IF INCORRECT!!!!!!!
            # else:
            #     gradient_u = np.vstack((0.0 * u[0].T, 0.2 * u[1].T))
            #     hess_u = [np.zeros((2,2)), np.identity(2) * 0.2]
                
            
            #print("eig of hess_x is: ", np.linalg.eig(hess_x))
            #hess_u = [np.identity(2) * 0.01, np.identity(2) * 0.01]
            #print("hess_x is: ", hess_x)
            #print("hess_u is: ", hess_u)
            #print("grad_x is: ", grad_x)
            #print("gradent_u is: ", gradient_u)
            #print("cost is: ", cost)
                
        else:
            #hess_x = np.identity(len(x))
            #grad_x = np.zeros((len(x), 1))
            #hess_u = [np.identity(2), np.identity(2)]
            #cost = 0.0
            
            #hess_x = np.identity(len(x)) * 0.05
            #grad_x = np.zeros((len(x), 1))
            #hess_u = [np.identity(2) * 0.05, np.identity(2) * 0.05]
            #cost = 0.0
            x_torch = torch.from_numpy(x).requires_grad_(True)
            u_torch = [torch.from_numpy(ui).requires_grad_(True) for ui in u]

            # Evaluate cost here.
            cost_torch = self.__call__(x_torch, u_torch, k, calc_deriv_cost)
            cost = cost_torch.item()
            
            # hess_x = np.identity(len(x)) * 0.0
            # grad_x = np.zeros((len(x), 1))
            # hess_u = [np.identity(2) * 0.1, np.identity(2) * 0.01]
            # cost = 0.0
            
            #hess_x = np.identity(len(x)) * 0.0
            hess_x = np.identity(len(x)) * 0.1739 # Change back to 0.01
            grad_x = np.zeros((len(x), 1))
            hess_u = [np.identity(2) * 0.1, np.identity(2) * 0.1]
            
            #gradient_u = np.zeros((2,2))
            #gradient_u[0,:] = 0.3 * u[0].T
            #gradient_u[1,:] = 0.3 * u[1].T
            gradient_u = np.vstack((0.1 * u[0].T, 0.1 * u[1].T))
            #cost = 0.0
            
            
            
            # if ii == 0:
            #     gradient_u = 0.2 * u[0].T
            #     hess_u = [np.identity(2) * 0.2]       #TRYING SOMETHING OUT HERE. DELETE IF INCORRECT!!!!!!!
            # else:
            #     gradient_u = 0.2 * u[1].T
            #     hess_u = [np.identity(2) * 0.2]
                
            
            # if ii == 0:
            #     gradient_u = np.vstack((0.2 * u[0].T, 0.0 * u[1].T))
            #     hess_u = [np.identity(2) * 0.2, np.zeros((2,2))]       #TRYING SOMETHING OUT HERE. DELETE IF INCORRECT!!!!!!!
            # else:
            #     gradient_u = np.vstack((0.0 * u[0].T, 0.2 * u[1].T))
            #     hess_u = [np.zeros((2,2)), np.identity(2) * 0.2]
            
            
        # Printing stuff
        #print("hess_x is: ", hess_x)
        #print("grad_x is: ", grad_x)
        #print("hess_u is: ", hess_u)
        #print("grad_u is: ", gradient_u)
        #print("hess_x[0] is: ", hess_x[0])
        #print("size of hess_x is: ", hess_x.shape)
        #print("eig of hess_x is: ", np.linalg.eig(hess_x))
        #print("hess_u is: ", hess_u)
            

        return cost, gradient_u, grad_x, hess_x, hess_u
