from cost.proximity_cost_reach_avoid_twoplayer import ProximityCost
from cost.obstacle_penalty import ObstacleDistCost
import torch
import numpy as np
import math
from torch.autograd import grad
from resource.car_5d import Car5Dv2

class ProximityCostTest():
    def __init__(self, config):
        self.config = config
        self.cost = ProximityCost(
            self.config["g_params"]["car"]["position_indices"][0],
            self.config["l_params"]["car"]["goals"][0],
            self.config["l_params"]["car"]["goal_radii"][0],
            name = "car_goal"
        )
    
    def compare_first_order(self, x, k=0):
        # calculate first order using manual calculation
        grad_manual = self.cost.first_order(x)
        # calculate first order using autograd
        x_torch = torch.from_numpy(x).requires_grad_(True)
        output = self.cost(x_torch)
        grad_autograd = torch.autograd.grad(outputs=output, inputs=x_torch, create_graph=True, allow_unused=True)[0]
        grad_autograd = grad_autograd.detach().cpu().numpy().flatten()

        result = np.array_equal(grad_autograd.astype(np.float16), grad_manual.astype(np.float16))

        if not result:
            print("Manual calc: \n{}".format(grad_manual))
            print("Autograd: \n{}".format(grad_autograd))

        return result

    def compare_second_order(self, x, k=0):
        # calculate second order using manual calculation
        hess_x_manual = self.cost.second_order(x)
        # calculate second order using autograd
        x_torch = torch.from_numpy(x).requires_grad_(True)
        output = self.cost(x_torch)
        grad_sample = torch.autograd.grad(outputs=output, inputs=x_torch, create_graph=True, allow_unused=True)[0]

        hess_x_autograd = np.zeros((len(x), len(x)))
        if grad_sample is not None and grad_sample.grad_fn is not None:
            print("First order is not None, calculating second order")
            for ii in range(len(x)):
                hess_row = torch.autograd.grad(
                    grad_sample[ii, 0], x_torch, retain_graph=True)[0]
                hess_x_autograd[ii, :] = hess_row.detach().numpy().copy().T
        
        result = np.array_equal(hess_x_autograd.astype(np.float16), hess_x_manual.astype(np.float16))

        if not result:
            print("Manual calc: \n{}".format(hess_x_manual))
            print("Autograd: \n{}".format(hess_x_autograd))

        return result

class ObstacleDistCostTest():
    def __init__(self, config):
        self.config = config
        self.cost = ObstacleDistCost(config["g_params"]["car"])
    
    def compare_first_order(self, x, k=0):
        # calculate first order using manual calculation
        grad_manual = self.cost.first_order(x)
        # calculate first order using autograd
        x_torch = torch.from_numpy(x).requires_grad_(True)
        output, func_output = self.cost(x_torch)
        grad_autograd = torch.autograd.grad(outputs=output, inputs=x_torch, create_graph=True, allow_unused=True)[0]
        grad_autograd = grad_autograd.detach().cpu().numpy().flatten()

        result = np.array_equal(grad_autograd.astype(np.float16), grad_manual.astype(np.float16))

        if not result:
            print("Manual calc: \n{}".format(grad_manual))
            print("Autograd: \n{}".format(grad_autograd))

        return result

    def compare_second_order(self, x, k=0):
        # calculate second order using manual calculation
        hess_x_manual = self.cost.second_order(x)
        # calculate second order using autograd
        x_torch = torch.from_numpy(x).requires_grad_(True)
        output, func_output = self.cost(x_torch)
        grad_sample = torch.autograd.grad(outputs=output, inputs=x_torch, create_graph=True, allow_unused=True)[0]

        hess_x_autograd = np.zeros((len(x), len(x)))
        if grad_sample is not None and grad_sample.grad_fn is not None:
            print("First order is not None, calculating second order")
            for ii in range(len(x)):
                hess_row = torch.autograd.grad(
                    grad_sample[ii, 0], x_torch, retain_graph=True)[0]
                hess_x_autograd[ii, :] = hess_row.detach().numpy().copy().T
        
        result = np.array_equal(hess_x_autograd.astype(np.float16), hess_x_manual.astype(np.float16))

        if not result:
            print("Manual calc: \n{}".format(hess_x_manual))
            print("Autograd: \n{}".format(hess_x_autograd))

        return result
