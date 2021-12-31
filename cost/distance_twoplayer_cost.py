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
# Proximity cost for state spaces that are Cartesian products of individual
# systems' state spaces. Penalizes
#      ``` sum_{i \ne j} min(distance(i, j) - max_distance, 0)^2 ```
# for all players i, j.
#
################################################################################

import torch
import numpy as np
import math

from cost.cost import Cost
from utils import MaxFuncMux

class ProductStateProximityCost(Cost):
    # def __init__(self, position_indices, max_distance, player_id, name=""):
    #     """
    #     Initialize with dimension to add cost to and threshold BELOW which
    #     to impose quadratic cost.

    #     :param position_indices: list of index tuples corresponding to (x, y)
    #     :type position_indices: [(uint, uint)]
    #     :param max_distance: maximum value of distance to penalize
    #     :type max_distance: float
    #     """
    #     self._position_indices = position_indices
    #     self._max_distance = max_distance
    #     self._num_players = len(position_indices)
    #     self._player_id = int(player_id)
    #     super(ProductStateProximityCost, self).__init__(name)

    def __init__(self, g_params, name="proximity"):
        """
        Initialize with dimension to add cost to and threshold BELOW which
        to impose quadratic cost.

        :param position_indices: list of index tuples corresponding to (x, y)
        :type position_indices: [(uint, uint)]
        :param max_distance: maximum value of distance to penalize
        :type max_distance: float
        """
        self._position_indices = g_params["position_indices"]
        self._max_distance = g_params["distance_threshold"]
        self._num_players = len(self._position_indices)
        self._player_id = int(g_params["player_id"])
        super(ProductStateProximityCost, self).__init__("car{}_".format(self._player_id+1)+name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.

        :param x: concatenated state vector of all systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        total_cost = torch.zeros(1, 1, requires_grad=True).double()
        # #print("Player id is: ", self._player_id)
        
        xi_idx, yi_idx = self._position_indices[self._player_id]
        
        for jj in range(self._num_players):
            if self._player_id == jj:
                continue
            
            # Compute relative distance
            xj_idx, yj_idx = self._position_indices[jj]
            dx = x[xi_idx, 0] - x[xj_idx, 0]
            dy = x[yi_idx, 0] - x[yj_idx, 0]

            relative_distance = torch.sqrt(torch.tensor(dx*dx + dy*dy))
            #total_cost += min(relative_distance - self._max_distance, 0.0)**2
        
        return (self._max_distance - relative_distance) * torch.ones(1, 1, requires_grad=True).double()
            

        # for ii in range(self._num_players):
        #     xi_idx, yi_idx = self._position_indices[ii]

        #     for jj in range(self._num_players):
        #         if ii == jj:
        #             continue

        #         # Compute relative distance.
        #         xj_idx, yj_idx = self._position_indices[jj]
        #         dx = x[xi_idx, 0] - x[xj_idx, 0]
        #         dy = x[yi_idx, 0] - x[yj_idx, 0]
        #         relative_distance = torch.sqrt(dx*dx + dy*dy)

        #         total_cost += min(
        #             relative_distance - self._max_distance, 0.0)**2

        #return total_cost

class CollisionPenalty(Cost):
    def __init__(self, g_params, name="collision_penalty"):
      self._position_indices = g_params["position_indices"]
      self._collision_r = g_params["collision_r"]
      self._car_params = g_params["car_params"]
      self._theta_indices = g_params["theta_indices"]
      self._player_id = g_params["player_id"]

      super(CollisionPenalty, self).__init__("car{}_".format(g_params["player_id"]+1)+name)
        
    # def get_car_state(self, x, index):
    #   car_x_index, car_y_index = self._position_indices[index]
    #   if type(x) is torch.Tensor:
    #     car_rear = x[car_x_index:car_x_index+2, 0]
    #     car_front = car_rear.add(torch.tensor([self._car_params["wheelbase"]*torch.cos(x[self._theta_indices[index], 0]), self._car_params["wheelbase"]*torch.sin(x[self._theta_indices[index], 0])], requires_grad=True))
    #   else:
    #     car_rear = np.array([x[car_x_index, 0], x[car_y_index, 0]])
    #     car_front = car_rear + np.array([self._car_params["wheelbase"]*math.cos(x[self._theta_indices[index], 0]), self._car_params["wheelbase"]*math.sin(x[self._theta_indices[index], 0])])
    #   return car_rear, car_front

    def get_car_state(self, x, index):
      car_x_index, car_y_index = self._position_indices[index]
      if type(x) is torch.Tensor:
          car_rear = x[car_x_index:2+car_x_index, 0]
          car_front = [
              x[car_x_index, 0] + self._car_params["wheelbase"]*torch.cos(x[self._theta_indices[index], 0]),
              x[car_y_index, 0] + self._car_params["wheelbase"]*torch.sin(x[self._theta_indices[index], 0])
          ]
      else:
          car_rear = np.array([x[car_x_index, 0], x[car_y_index, 0]])
          car_front = [
              x[car_x_index, 0] + self._car_params["wheelbase"]*math.cos(x[self._theta_indices[index], 0]),
              x[car_y_index, 0] + self._car_params["wheelbase"]*math.sin(x[self._theta_indices[index], 0])
          ]
      return car_rear, car_front

    def g_coll_ff(self, x, k = 0, **kwargs):
      _car1_rear, _car1_front = self.get_car_state(x, self._player_id)
      _car2_rear, _car2_front = self.get_car_state(x, 1-self._player_id)
      if type(x) is torch.Tensor:
        return 2.0 * self._collision_r - torch.sqrt((_car1_front[0] - _car2_front[0]) ** 2 + (_car1_front[1] - _car2_front[1]) ** 2)
      else:
        return 2.0 * self._collision_r - math.sqrt((_car1_front[0] - _car2_front[0]) ** 2 + (_car1_front[1] - _car2_front[1]) ** 2)

    def g_coll_fr(self, x, k = 0, **kwargs):
      _car1_rear, _car1_front = self.get_car_state(x, self._player_id)
      _car2_rear, _car2_front = self.get_car_state(x, 1-self._player_id)
      if type(x) is torch.Tensor:
        return 2.0 * self._collision_r - torch.sqrt((_car1_front[0] - _car2_rear[0]) ** 2 + (_car1_front[1] - _car2_rear[1]) ** 2)
      else:
        return 2.0 * self._collision_r - math.sqrt((_car1_front[0] - _car2_rear[0]) ** 2 + (_car1_front[1] - _car2_rear[1]) ** 2)

    def g_coll_rf(self, x, k = 0, **kwargs):
      _car1_rear, _car1_front = self.get_car_state(x, self._player_id)
      _car2_rear, _car2_front = self.get_car_state(x, 1-self._player_id)
      if type(x) is torch.Tensor:
        return 2.0 * self._collision_r - torch.sqrt((_car1_rear[0] - _car2_front[0]) ** 2 + (_car1_rear[1] - _car2_front[1]) ** 2)
      else:
        return 2.0 * self._collision_r - math.sqrt((_car1_rear[0] - _car2_front[0]) ** 2 + (_car1_rear[1] - _car2_front[1]) ** 2)

    def g_coll_rr(self, x, k = 0, **kwargs):
      _car1_rear, _car1_front = self.get_car_state(x, self._player_id)
      _car2_rear, _car2_front = self.get_car_state(x, 1-self._player_id)
      if type(x) is torch.Tensor:
        return 2.0 * self._collision_r - torch.sqrt((_car1_rear[0] - _car2_rear[0]) ** 2 + (_car1_rear[1] - _car2_rear[1]) ** 2)
      else:
        return 2.0 * self._collision_r - math.sqrt((_car1_rear[0] - _car2_rear[0]) ** 2 + (_car1_rear[1] - _car2_rear[1]) ** 2)

    def g_collision(self, x, **kwargs):
      _max_func = MaxFuncMux()
      _max_func.store(self.g_coll_ff, self.g_coll_ff(x))
      _max_func.store(self.g_coll_fr, self.g_coll_fr(x))
      _max_func.store(self.g_coll_rf, self.g_coll_rf(x))
      _max_func.store(self.g_coll_rr, self.g_coll_rr(x))
      func_of_max_val, max_val = _max_func.get_max()
      return max_val, func_of_max_val

    # def g_rearonly_collision(self, x, k=0, **kwargs):
    #   car1_x_index, car1_y_index = self._position_indices[0]
    #   car2_x_index, car2_y_index = self._position_indices[1]
    #   if type(x) is torch.Tensor:
    #     return 2.0 * self._collision_r - torch.sqrt((x[car1_x_index, 0] - x[car2_x_index, 0]) ** 2 + (x[car1_y_index, 0] - x[car2_y_index, 0]) ** 2)
    #   else:
    #     return 2.0 * self._collision_r - math.sqrt((x[car1_x_index, 0] - x[car2_x_index, 0]) ** 2 + (x[car1_y_index, 0] - x[car2_y_index, 0]) ** 2)


    # def g_collision(self, x, **kwargs):
    #   return self.g_rearonly_collision(x), self.g_rearonly_collision

    def __call__(self, x, k=0):
      max_val, func_of_max_val = self.g_collision(x)
      return max_val, func_of_max_val

class PedestrianToCarCollisionPenalty(Cost):
    """
    Passing in g_params with car information: g_params["car1"]
    """
    def __init__(self, g_params, name="collision_penalty"):
      self._position_indices = g_params["position_indices"]
      self._collision_r = g_params["collision_r"]
      self._car_params = g_params["car_params"]
      self._theta_indices = g_params["theta_indices"]
      # WARNING: THIS IS CAR ID, NOT PEDESTRIAN ID
      self._player_id = g_params["player_id"]

      super(PedestrianToCarCollisionPenalty, self).__init__("ped_car{}_".format(g_params["player_id"]+1)+name)

    def get_car_state(self, x, index):
      car_x_index, car_y_index = self._position_indices[index]
      if type(x) is torch.Tensor:
          car_rear = x[car_x_index:2+car_x_index, 0]
          car_front = [
              x[car_x_index, 0] + self._car_params["wheelbase"]*torch.cos(x[self._theta_indices[index], 0]),
              x[car_y_index, 0] + self._car_params["wheelbase"]*torch.sin(x[self._theta_indices[index], 0])
          ]
      else:
          car_rear = np.array([x[car_x_index, 0], x[car_y_index, 0]])
          car_front = [
              x[car_x_index, 0] + self._car_params["wheelbase"]*math.cos(x[self._theta_indices[index], 0]),
              x[car_y_index, 0] + self._car_params["wheelbase"]*math.sin(x[self._theta_indices[index], 0])
          ]
      return car_rear, car_front

    def g_coll_f(self, x, k = 0, **kwargs):
      _car_rear, _car_front = self.get_car_state(x, self._player_id)
      ped_x_index, ped_y_index = self._position_indices[2]

      if type(x) is torch.Tensor:
        return self._collision_r - torch.sqrt((_car_front[0] - x[ped_x_index, 0]) ** 2 + (_car_front[1] - x[ped_y_index, 0]) ** 2)
      else:
        return self._collision_r - math.sqrt((_car_front[0] - x[ped_x_index, 0]) ** 2 + (_car_front[1] - x[ped_y_index, 0]) ** 2)

    def g_coll_r(self, x, k = 0, **kwargs):
      _car_rear, _car_front = self.get_car_state(x, self._player_id)
      ped_x_index, ped_y_index = self._position_indices[2]

      if type(x) is torch.Tensor:
        return self._collision_r - torch.sqrt((_car_rear[0] - x[ped_x_index, 0]) ** 2 + (_car_rear[1] - x[ped_y_index, 0]) ** 2)
      else:
        return self._collision_r - math.sqrt((_car_rear[0] - x[ped_x_index, 0]) ** 2 + (_car_rear[1] - x[ped_y_index, 0]) ** 2)

    def g_collision(self, x, **kwargs):
      _max_func = MaxFuncMux()
      _max_func.store(self.g_coll_f, self.g_coll_f(x))
      _max_func.store(self.g_coll_r, self.g_coll_r(x))
      func_of_max_val, max_val = _max_func.get_max()
      return max_val, func_of_max_val

    def __call__(self, x, k=0):
      max_val, func_of_max_val = self.g_collision(x)
      return max_val, func_of_max_val