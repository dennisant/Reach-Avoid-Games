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
# Proximity cost, derived from Cost base class. Implements a cost function that
# depends only on state and penalizes -min(distance, max_distance)^2.
#
################################################################################

from typing import Type
import torch
import numpy as np
import matplotlib.pyplot as plt

from cost import Cost
from point import Point

class ProximityCost(Cost):
    def __init__(self, position_indices, point,
                max_distance, outside_weight=0.1, apply_after_time=1,
                name=""): # Maybe change back to apply_after_time=20
      """
      Initialize with dimension to add cost to and threshold BELOW which
      to impose quadratic cost. Above the threshold, we use a very light
      quadratic cost. The overall cost is continuous.

      :param position_indices: indices of input corresponding to (x, y)
      :type position_indices: (uint, uint)
      :param point: point from which to compute proximity
      :type point: Point
      :param max_distance: maximum value of distance to penalize
      :type threshold: float
      :param outside_weight: weight of quadratic cost outside threshold
      :type outside_weight: float
      :param apply_after_time: only apply proximity time after this time step
      :type apply_after_time: int
      """
      self._x_index, self._y_index = position_indices
      self._point = point
      #self._max_squared_distance = max_distance**2
      self._max_squared_distance = max_distance ** 2
      self._max_distance = max_distance
      self._outside_weight = outside_weight
      self._apply_after_time = apply_after_time
      super(ProximityCost, self).__init__(name)

    def __call__(self, x, k=0):
      """
      Evaluate this cost function on the given input state and time.
      NOTE: `x` should be a column vector.

      :param x: concatenated state of the two systems
      :type x: torch.Tensor
      :param k: time step, if cost is time-varying
      :type k: uint
      :return: scalar value of cost
      :rtype: torch.Tensor
      """
      # if k < self._apply_after_time:
      #     return torch.zeros(
      #         1, 1, requires_grad=True).double()
      #print("k is: ", k)
      
      #else:
      # Compute relative distance.
      #print("Goal x-position is: ", self._point.x)
      #print("Goal y-position is: ", self._point.y)
      #print("x-position is: ", x[self._x_index, 0])
      #print("y-position is: ", x[self._y_index, 0])
      dx = x[self._x_index, 0] - self._point[0]
      dy = x[self._y_index, 0] - self._point[1]
      relative_squared_distance = torch.sqrt(dx*dx + dy*dy) # Comment out the original one below
      #relative_squared_distance = dx*dx + dy*dy
      
      return (relative_squared_distance - self._max_distance) * torch.ones(1, 1, requires_grad=True).double()
      #return relative_squared_distance * torch.ones(1, 1, requires_grad=True).double()

      # # Compute relative distance.
      # dx = x[self._x_index, 0] - self._point.x
      # dy = x[self._y_index, 0] - self._point.y
      # relative_squared_distance = dx*dx + dy*dy

      # if relative_squared_distance < self._max_squared_distance:
      #     return -relative_squared_distance * torch.ones(
      #         1, 1, requires_grad=True).double()

      # # Outside penalty is:
      # #   ``` outside_weight * (relative_distance - max_distance)**2 ```
      # # which can be computed from what we have already with only one sqrt.
      # outside_penalty = self._outside_weight * (
      #     relative_squared_distance + self._max_squared_distance -
      #     2.0 * torch.sqrt(
      #         relative_squared_distance * self._max_squared_distance))
      # return -outside_penalty - self._max_squared_distance * torch.ones(
      #     1, 1, requires_grad=True).double()

    def render(self, ax=None):
      """ Render this obstacle on the given axes. """
      if np.isinf(self._max_squared_distance):
          radius = 1.0 # 1.0
      else:
          radius = np.sqrt(self._max_squared_distance)
      circle = plt.Circle(
          (self._point.x, self._point.y), radius,
          color="g", fill=True, alpha=0.5)
      ax.add_artist(circle)
      ax.text(self._point.x, self._point.y, "goal", fontsize=10)

class ProximityCostDuo(Cost):
    def __init__(self, player_id, position_indices, point,
                max_distance, outside_weight=0.1, apply_after_time=1,
                name=""): # Maybe change back to apply_after_time=20
      """
      Initialize with dimension to add cost to and threshold BELOW which
      to impose quadratic cost. Above the threshold, we use a very light
      quadratic cost. The overall cost is continuous.

      :param position_indices: indices of input corresponding to (x, y)
      :type position_indices: (uint, uint)
      :param point: point from which to compute proximity
      :type point: Point
      :param max_distance: maximum value of distance to penalize
      :type threshold: float
      :param outside_weight: weight of quadratic cost outside threshold
      :type outside_weight: float
      :param apply_after_time: only apply proximity time after this time step
      :type apply_after_time: int
      """
      self._x_index, self._y_index = position_indices
      self._point_1, self._point_2 = point
      self._max_squared_distance = max_distance ** 2
      self._max_distance = max_distance
      self._outside_weight = outside_weight
      self._apply_after_time = apply_after_time
      self._player_id = player_id
      super(ProximityCostDuo, self).__init__(name)

    def __call__(self, x, k=0):
      """
      Evaluate this cost function on the given input state and time.
      NOTE: `x` should be a column vector.

      :param x: concatenated state of the two systems
      :type x: torch.Tensor
      :param k: time step, if cost is time-varying
      :type k: uint
      :return: scalar value of cost
      :rtype: torch.Tensor
      """
      dx = x[self._x_index, 0] - self._point_1[0]
      dy = x[self._y_index, 0] - self._point_1[1]
      relative_squared_distance_1 = torch.sqrt(dx*dx + dy*dy) # Comment out the original one below

      dx = x[self._x_index, 0] - self._point_2[0]
      dy = x[self._y_index, 0] - self._point_2[1]
      relative_squared_distance_2 = torch.sqrt(dx*dx + dy*dy) # Comment out the original one below
      
      return torch.min(
        relative_squared_distance_1 - self._max_distance,
        relative_squared_distance_2 - self._max_distance) * torch.ones(1, 1, requires_grad=True).double()

    def render(self, ax=None):
      """ Render this obstacle on the given axes. """
      if np.isinf(self._max_squared_distance):
          radius = 1.0 # 1.0
      else:
          radius = np.sqrt(self._max_squared_distance)

      if self._player_id == 0:
        color = "r"
      else:
        color = "g"

      circle_1 = plt.Circle(
          (self._point_1.x, self._point_1.y), radius,
          color=color, fill=True, alpha=0.5)
      circle_2 = plt.Circle(
          (self._point_2.x, self._point_2.y), radius,
          color=color, fill=True, alpha=0.5)
      ax.add_artist(circle_1)
      ax.add_artist(circle_2)
      ax.text(self._point_1.x, self._point_1.y, "goal", fontsize=10)
      ax.text(self._point_2.x, self._point_2.y, "goal", fontsize=10)

class ProximityToBlockCost(Cost):
    def __init__(self, g_params, name=""):
      self._x_index, self._y_index = g_params["position_indices"][g_params["player_id"]]
      self._goal_x, self._goal_y = g_params["goals"]
      self._player_id = g_params["player_id"]
      self._road_rules = g_params["road_rules"]
      self._road_logic = self.get_road_logic_dict(g_params["road_logic"])
      self._road_rules = self.new_road_rules()

      super(ProximityToBlockCost, self).__init__(name)

    def get_road_logic_dict(self, road_logic):
      return {
        "left_lane": road_logic[0] == 1, 
        "right_lane": road_logic[1] == 1, 
        "up_lane": road_logic[2] == 1, 
        "down_lane": road_logic[3] == 1, 
        "left_turn": road_logic[4] == 1
      }

    def __call__(self, x, k=0):
        if self._player_id == 0:
            # dy = self._goal_y - x[self._y_index, 0]
            # return torch.tensor(dy) * torch.ones(1, 1, requires_grad=True).double()
            if type(x) is torch.Tensor:
              value = torch.max(
                  abs(self._goal_y - x[self._y_index, 0]) * (self._goal_y - x[self._y_index, 0]),
                  torch.max(
                    abs(x[self._x_index, 0] - self._road_rules["x_max"]) * (x[self._x_index, 0] - self._road_rules["x_max"]),
                    abs(self._road_rules["x_min"] - x[self._x_index, 0]) * (self._road_rules["x_min"] - x[self._x_index, 0])
                  )
                )
              # value = torch.min(
              #   torch.max(
              #     abs(self._goal_x - x[self._x_index, 0]) * (self._goal_x - x[self._x_index, 0]),
              #     torch.max(
              #       abs(x[self._y_index, 0] - self._road_rules["y_max"]) * (x[self._y_index, 0] - self._road_rules["y_max"]),
              #       abs(self._road_rules["y_min"] - x[self._y_index, 0]) * (self._road_rules["y_min"] - x[self._y_index, 0])
              #     )
              #   ),
              #   torch.max(
              #     abs(self._goal_y - x[self._y_index, 0]) * (self._goal_y - x[self._y_index, 0]),
              #     torch.max(
              #       abs(x[self._x_index, 0] - self._road_rules["x_max"]) * (x[self._x_index, 0] - self._road_rules["x_max"]),
              #       abs(self._road_rules["x_min"] - x[self._x_index, 0]) * (self._road_rules["x_min"] - x[self._x_index, 0])
              #     )
              #   )
              # )
            else:
              value = max(
                  abs(self._goal_y - x[self._y_index, 0]) * (self._goal_y - x[self._y_index, 0]),
                  max(
                    abs(x[self._x_index, 0] - self._road_rules["x_max"]) * (x[self._x_index, 0] - self._road_rules["x_max"]),
                    abs(self._road_rules["x_min"] - x[self._x_index, 0]) * (self._road_rules["x_min"] - x[self._x_index, 0])
                  )
                )
              # value = min(
              #   max(
              #     abs(self._goal_x - x[self._x_index, 0]) * (self._goal_x - x[self._x_index, 0]),
              #     max(
              #       abs(x[self._y_index, 0] - self._road_rules["y_max"]) * (x[self._y_index, 0] - self._road_rules["y_max"]),
              #       abs(self._road_rules["y_min"] - x[self._y_index, 0]) * (self._road_rules["y_min"] - x[self._y_index, 0])
              #     )
              #   ),
              #   max(
              #     abs(self._goal_y - x[self._y_index, 0]) * (self._goal_y - x[self._y_index, 0]),
              #     max(
              #       abs(x[self._x_index, 0] - self._road_rules["x_max"]) * (x[self._x_index, 0] - self._road_rules["x_max"]),
              #       abs(self._road_rules["x_min"] - x[self._x_index, 0]) * (self._road_rules["x_min"] - x[self._x_index, 0])
              #     )
              #   )
              # )
            return value
        else:
            if type(x) is torch.Tensor:
              value = torch.max(
                  abs(self._goal_x - x[self._x_index, 0]) * (self._goal_x - x[self._x_index, 0]),
                  torch.max(
                    abs(x[self._y_index, 0] - self._road_rules["y_max"]) * (x[self._y_index, 0] - self._road_rules["y_max"]),
                    abs(self._road_rules["y_min"] - x[self._y_index, 0]) * (self._road_rules["y_min"] - x[self._y_index, 0])
                  )
                )
              # value = torch.min(
              #   torch.max(
              #     abs(self._goal_x - x[self._x_index, 0]) * (self._goal_x - x[self._x_index, 0]),
              #     torch.max(
              #       abs(x[self._y_index, 0] - self._road_rules["y_max"]) * (x[self._y_index, 0] - self._road_rules["y_max"]),
              #       abs(self._road_rules["y_min"] - x[self._y_index, 0]) * (self._road_rules["y_min"] - x[self._y_index, 0])
              #     )
              #   ),
              #   torch.max(
              #     abs(x[self._y_index, 0] - self._goal_y) * (x[self._y_index, 0] - self._goal_y),
              #     torch.max(
              #       abs(x[self._x_index, 0] - self._road_rules["x_max"]) * (x[self._x_index, 0] - self._road_rules["x_max"]),
              #       abs(self._road_rules["x_min"] - x[self._x_index, 0]) * (self._road_rules["x_min"] - x[self._x_index, 0])
              #     )
              #   ),
              # )
            else:
              value = max(
                  abs(self._goal_x - x[self._x_index, 0]) * (self._goal_x - x[self._x_index, 0]),
                  max(
                    abs(x[self._y_index, 0] - self._road_rules["y_max"]) * (x[self._y_index, 0] - self._road_rules["y_max"]),
                    abs(self._road_rules["y_min"] - x[self._y_index, 0]) * (self._road_rules["y_min"] - x[self._y_index, 0])
                  )
                )
              # value = min(
              #   max(
              #     abs(self._goal_x - x[self._x_index, 0]) * (self._goal_x - x[self._x_index, 0]),
              #     max(
              #       abs(x[self._y_index, 0] - self._road_rules["y_max"]) * (x[self._y_index, 0] - self._road_rules["y_max"]),
              #       abs(self._road_rules["y_min"] - x[self._y_index, 0]) * (self._road_rules["y_min"] - x[self._y_index, 0])
              #     )
              #   ),
              #   max(
              #     abs(x[self._y_index, 0] - self._goal_y) * (x[self._y_index, 0] - self._goal_y),
              #     max(
              #       abs(x[self._x_index, 0] - self._road_rules["x_max"]) * (x[self._x_index, 0] - self._road_rules["x_max"]),
              #       abs(self._road_rules["x_min"] - x[self._x_index, 0]) * (self._road_rules["x_min"] - x[self._x_index, 0])
              #     )
              #   ),
              # )
            return value

    def render(self, ax=None, contour = False, player=0):
        """ Render this obstacle on the given axes. """
        if self._player_id == 0:
          ax.plot([self._road_rules["x_min"], self._road_rules["x_max"]], [self._goal_y, self._goal_y], c = 'r', linewidth = 10, alpha = 0.2)
          # ax.plot([self._goal_x, self._goal_x], [self._road_rules["y_min"], self._road_rules["y_max"]], c = 'r', linewidth = 10, alpha = 0.2)
        else:
          ax.plot([self._goal_x, self._goal_x], [self._road_rules["y_min"], self._road_rules["y_max"]], c = 'g', linewidth = 10, alpha = 0.2)
          # ax.plot([self._road_rules["x_min"], self._road_rules["x_max"]], [self._goal_y, self._goal_y], c = 'g', linewidth = 10, alpha = 0.2)

        if contour and self._player_id == player:
          self.target_contour(ax)
    
    def target_contour(self, ax=None):
        x_range = np.arange(0, 25, step = 0.1)
        y_range = np.arange(0, 30, step = 0.1)
        zz = np.array([[0]*250]*300)
        for x in x_range:
            for y in y_range:
                xs = np.array([x, y, 0, 0, 0, x, y, 0, 0, 0]).reshape(10, 1)
                zz[int(y*10)][int(x*10)] = self(xs)
        contour = ax.contourf(x_range, y_range, zz, cmap = "YlGn", alpha = 0.5, levels = np.arange(-10, 20, step=1))
        plt.colorbar(contour)
    
    def new_road_rules(self, **kwargs):
      import copy

      left_lane = self._road_logic["left_lane"]
      right_lane = self._road_logic["right_lane"]
      down_lane = self._road_logic["down_lane"]
      up_lane = self._road_logic["up_lane"]

      for key in kwargs.keys():
        if key == "left_lane":
          left_lane = kwargs["left_lane"]
        if key == "right_lane":
          right_lane = kwargs["right_lane"]
        if key == "down_lane":
          down_lane = kwargs["down_lane"]
        if key == "up_lane":
          up_lane = kwargs["up_lane"]

      new_road_rules = copy.deepcopy(self._road_rules)

      if down_lane and not up_lane:
        new_road_rules["y_max"] = self._road_rules["y_max"] - self._road_rules["width"]
      elif up_lane and not down_lane:
        new_road_rules["y_min"] = self._road_rules["y_min"] + self._road_rules["width"]
      
      if left_lane and not right_lane:
        # Can either go straight down or turn left
        new_road_rules["x_max"] = self._road_rules["x_max"] - self._road_rules["width"]
      elif right_lane and not left_lane:
        # Can either go straight up or turn right
        new_road_rules["x_min"] = self._road_rules["x_min"] + self._road_rules["width"]

      return new_road_rules