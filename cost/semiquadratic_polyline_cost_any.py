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
# Semiquadratic cost that takes effect a fixed distance away from a Polyline.
#
################################################################################

import torch
import matplotlib.pyplot as plt

from cost.cost import Cost
from resource.point import Point
from resource.polyline import Polyline
from utils.utils import MaxFuncMux, MinFuncMux
import numpy as np
import math

class SemiquadraticPolylineCostAny(Cost):
    # def __init__(self, polyline, distance_threshold, position_indices, name=""):
    #     """
    #     Initialize with a polyline, a threshold in distance from the polyline.
    #     :param polyline: piecewise linear path which defines signed distances
    #     :type polyline: Polyline
    #     :param distance_threshold: value above which to penalize
    #     :type distance_threshold: float
    #     :param position_indices: indices of input corresponding to (x, y)
    #     :type position_indices: (uint, uint)
    #     """
    #     self._polyline = polyline
    #     self._distance_threshold = distance_threshold
    #     self._x_index, self._y_index = position_indices
    #     super(SemiquadraticPolylineCostAny, self).__init__(name)

    def __init__(self, g_params, name="polyline_boundary"):
        """
        Initialize with a polyline, a threshold in distance from the polyline.
        :param polyline: piecewise linear path which defines signed distances
        :type polyline: Polyline
        :param distance_threshold: value above which to penalize
        :type distance_threshold: float
        :param position_indices: indices of input corresponding to (x, y)
        :type position_indices: (uint, uint)
        """
        self._polyline = g_params["polyline"]
        self._distance_threshold = g_params["lane_width"]
        self._x_index, self._y_index = g_params["position_indices"][g_params["player_id"]]
        super(SemiquadraticPolylineCostAny, self).__init__("car{}_".format(g_params["player_id"]+1)+name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given state and time.
        NOTE: `x` should be a PyTorch tensor with `requires_grad` set `True`.
        NOTE: `x` should be a column vector.
        :param x: state of the system
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        signed_distance = self._polyline.signed_distance_to(
            Point(x[self._x_index, 0], x[self._y_index, 0]))
        
        #print("(abs(signed_distance) - self._distance_threshold) is: ", abs(signed_distance) - self._distance_threshold)
        return (abs(signed_distance) - self._distance_threshold) * torch.ones(1, 1, requires_grad=True).double()

    def render(self, ax=None):
        """ Render this cost on the given axes. """
        xs = [pt.x for pt in self._polyline.points]
        ys = [pt.y for pt in self._polyline.points]
        ax.plot(xs, ys, "k", alpha=0.25)

class RoadRulesPenalty(Cost):
    def __init__(self, g_params, name="roadrules_penalty"):
        """
        Penalty based on road rules, takes in g_params with position_indices, player_id, road_rules
        """
        self._x_index, self._y_index = g_params["position_indices"][g_params["player_id"]]
        
        self._road_rules_original = g_params["road_rules"]
        self._road_logic = self.get_road_logic_dict(g_params["road_logic"])
        self._road_rules = self.new_road_rules()
        self._collision_r = g_params["collision_r"]
        self._collision_r = 0
        self._player_id = g_params["player_id"]
        self._car_params = g_params["car_params"]
        self._theta_indices = g_params["theta_indices"]
        super(RoadRulesPenalty, self).__init__("car{}_".format(g_params["player_id"]+1)+name)
    
    # def get_car_state(self, x, index):
    #   car_x_index, car_y_index = self._x_index, self._y_index
    #   if type(x) is torch.Tensor:
    #     car_rear = x[car_x_index:car_x_index+2, 0]
    #     car_front = car_rear.add(torch.tensor([self._car_params["wheelbase"]*torch.cos(x[self._theta_indices[index], 0]), self._car_params["wheelbase"]*torch.sin(x[self._theta_indices[index], 0])], requires_grad=True))
    #   else:
    #     car_rear = np.array([x[car_x_index, 0], x[car_y_index, 0]])
    #     car_front = car_rear + np.array([self._car_params["wheelbase"]*math.cos(x[self._theta_indices[index], 0]), self._car_params["wheelbase"]*math.sin(x[self._theta_indices[index], 0])])
    #   return car_rear, car_front

    def get_car_state(self, x, index):
      car_x_index, car_y_index = self._x_index, self._y_index
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

    def get_road_logic_dict(self, road_logic):
      return {
        "left_lane": road_logic[0] == 1, 
        "right_lane": road_logic[1] == 1, 
        "up_lane": road_logic[2] == 1, 
        "down_lane": road_logic[3] == 1, 
        "left_turn": road_logic[4] == 1
      }

    def g_left_of_main(self, x, k=0, **kwargs):
      if "road_rules" in kwargs.keys():
        _road_rules = kwargs["road_rules"]
      else: 
        _road_rules = self._road_rules

      car_rear, car_front = self.get_car_state(x, self._player_id)
      if type(x) is torch.Tensor:
        return torch.max(
          _road_rules["x_min"] - car_rear[0] + self._collision_r,
          _road_rules["x_min"] - car_front[0] + self._collision_r
        )
      else:
        return max(
          _road_rules["x_min"] - car_rear[0] + self._collision_r,
          _road_rules["x_min"] - car_front[0] + self._collision_r
        )

    def g_right_of_main(self, x, k=0, **kwargs):
      if "road_rules" in kwargs.keys():
        _road_rules = kwargs["road_rules"]
      else: 
        _road_rules = self._road_rules
      
      car_rear, car_front = self.get_car_state(x, self._player_id)
      if type(x) is torch.Tensor:
        return torch.max(
          car_rear[0] - _road_rules["x_max"] + self._collision_r,
          car_front[0] - _road_rules["x_max"] + self._collision_r
        )
      else:
        return max(
          car_rear[0] - _road_rules["x_max"] + self._collision_r,
          car_front[0] - _road_rules["x_max"] + self._collision_r
        )

    def g_outside_rightband(self, x, k=0, **kwargs):
      if "road_rules" in kwargs.keys():
        _road_rules = kwargs["road_rules"]
      else: 
        _road_rules = self._road_rules

      car_rear, car_front = self.get_car_state(x, self._player_id)
      if type(x) is torch.Tensor:
        return torch.max(
          torch.max(
            car_rear[1] - _road_rules["y_max"] + self._collision_r,
            car_front[1] - _road_rules["y_max"] + self._collision_r
          ),
          torch.max(
            _road_rules["y_min"] - car_rear[1] + self._collision_r,
            _road_rules["y_min"] - car_front[1] + self._collision_r
          )
        )
      else:
        return max(
          max(
            car_rear[1] - _road_rules["y_max"] + self._collision_r,
            car_front[1] - _road_rules["y_max"] + self._collision_r
          ),
          max(
            _road_rules["y_min"] - car_rear[1] + self._collision_r,
            _road_rules["y_min"] - car_front[1] + self._collision_r
          )
        )

    def g_right_combined_withcurve(self, x, k=0):
      # first layer
      layer_1_road_rules = self.new_road_rules(left_lane = True, right_lane = False, up_lane = True, down_lane = True)
      # second layer
      layer_2_road_rules = self.new_road_rules(left_lane = True, right_lane = True, down_lane = True, up_lane = False)

      if type(x) is torch.Tensor:
        value = torch.max(
          torch.min(
            self.g_right_of_main(x, road_rules = layer_1_road_rules),
            self.g_outside_rightband(x, road_rules = layer_1_road_rules)
          ),
          torch.min(
            self.g_right_of_main(x, road_rules = layer_2_road_rules),
            self.g_outside_rightband(x, road_rules = layer_2_road_rules)
          )
        )
      else:
        value = max(
          min(
            self.g_right_of_main(x, road_rules = layer_1_road_rules),
            self.g_outside_rightband(x, road_rules = layer_1_road_rules)
          ),
          min(
            self.g_right_of_main(x, road_rules = layer_2_road_rules),
            self.g_outside_rightband(x, road_rules = layer_2_road_rules)
          )
        )   
      return value

    def g_right_combined(self, x):
      _min_func = MinFuncMux()
      _min_func.store(self.g_right_of_main, self.g_right_of_main(x))
      _min_func.store(self.g_outside_rightband, self.g_outside_rightband(x))
      return _min_func.get_min()

    def g_circle_intersection(self, x, k=0):
      _r = self._road_rules["width"]
      return _r**2 - (x[self._x_index, 0] - self._road_rules["x_max"] - _r) ** 2 - (x[self._y_index, 0] - self._road_rules["y_max"] - _r) ** 2

    def g_road_rules(self, x, **kwargs):
      left_turn = self._road_logic["left_turn"]

      for key in kwargs.keys():
        if key == "left_turn":
          left_turn = kwargs["left_turn"]

      _max_func = MaxFuncMux()
      if not left_turn:
          _max_func.store(self.g_left_of_main, self.g_left_of_main(x))
          _func_of_min_val, _min_val = self.g_right_combined(x)
          _max_func.store(_func_of_min_val, _min_val)
          _func_of_max_val, _max_val = _max_func.get_max()
      else:
          _max_func.store(self.g_left_of_main, self.g_left_of_main(x))
          _max_func.store(self.g_circle_intersection, self.g_circle_intersection(x))
          _max_func.store(self.g_right_combined_withcurve, self.g_right_combined_withcurve(x))
          _func_of_max_val, _max_val = _max_func.get_max()
      return _max_val, _func_of_max_val

    def __call__(self, x, k=0):
      _max_val, _func_of_max_val = self.g_road_rules(x)
      return _max_val, _func_of_max_val

    def render(self, ax=None):
      """ Render this cost on the given axes. """
      x_range = np.arange(0, 25, step = 0.1)
      y_range = np.arange(0, 40, step = 0.1)
      zz = np.array([[0]*250]*400)
      for x in x_range:
          for y in y_range:
              xs = np.array([x, y, np.pi * 0.5, 0, 0, x, y, -np.pi * 0.5, 0, 0]).reshape(10, 1)
              max_val, func_of_max_val = self(xs)
              zz[int(y*10)][int(x*10)] = max_val
      # contour = ax.contourf(x_range, y_range, zz, cmap = "YlGn", alpha = 0.5, levels = np.arange(-10, 30, step=2.5))
      contour = ax.contourf(x_range, y_range, zz, cmap = "Purples", alpha = 0.3, levels = [-3, -2, -1, 0], extend = "both")
      ax.clabel(contour, inline=True, fontsize=10, colors="k")
      contour.cmap.set_under('white')
      contour.cmap.set_over('navy')
      # plt.colorbar(contour)

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

      new_road_rules = copy.deepcopy(self._road_rules_original)

      if down_lane and not up_lane:
        new_road_rules["y_max"] = self._road_rules_original["y_max"] - self._road_rules_original["width"]
      elif up_lane and not down_lane:
        new_road_rules["y_min"] = self._road_rules_original["y_min"] + self._road_rules_original["width"]
      
      if left_lane and not right_lane:
        # Can either go straight down or turn left
        new_road_rules["x_max"] = self._road_rules_original["x_max"] - self._road_rules_original["width"]
      elif right_lane and not left_lane:
        # Can either go straight up or turn right
        new_road_rules["x_min"] = self._road_rules_original["x_min"] + self._road_rules_original["width"]

      return new_road_rules