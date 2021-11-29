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

from cost import Cost
from point import Point
from polyline import Polyline
from utils import MaxFuncMux, MinFuncMux

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
        
        self._road_rules = g_params["road_rules"]        
        self._road_logic = self.get_road_logic_dict(g_params["road_logic"])
        self._road_rules = self.new_road_rules()

        # if self._road_logic["down_lane"] and not self._road_logic["up_lane"]:
        #   self._road_rules["y_max"] = self._road_rules["y_max"] - self._road_rules["width"]
        # elif self._road_logic["up_lane"] and not self._road_logic["down_lane"]:
        #   self._road_rules["y_min"] = self._road_rules["y_min"] + self._road_rules["width"]
        
        # if self._road_logic["left_lane"] and not self._road_logic["right_lane"]:
        #   # Can either go straight down or turn left
        #   self._road_rules["x_max"] = self._road_rules["x_max"] - self._road_rules["width"]
        # elif self._road_logic["right_lane"] and not self._road_logic["left_lane"]:
        #   # Can either go straight up or turn right
        #   self._road_rules["x_min"] = self._road_rules["x_min"] + self._road_rules["width"]

        self._max_func = MaxFuncMux()
        super(RoadRulesPenalty, self).__init__("car{}_".format(g_params["player_id"]+1)+name)

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

      return torch.tensor(_road_rules["x_min"] - x[self._x_index, 0]) * torch.ones(1, 1, requires_grad=True).double()

    def g_right_of_main(self, x, k=0, **kwargs):
      if "road_rules" in kwargs.keys():
        _road_rules = kwargs["road_rules"]
      else: 
        _road_rules = self._road_rules
      
      return torch.tensor(x[self._x_index, 0] - _road_rules["x_max"]) * torch.ones(1, 1, requires_grad=True).double()

    def g_outside_rightband(self, x, k=0, **kwargs):
      if "road_rules" in kwargs.keys():
        _road_rules = kwargs["road_rules"]
      else: 
        _road_rules = self._road_rules

      return torch.max(
          torch.tensor(x[self._y_index, 0] - _road_rules["y_max"]),
          torch.tensor(_road_rules["y_min"] - x[self._y_index, 0])
      ) * torch.ones(1, 1, requires_grad=True).double()

    # def g_right_combined_withcurve(self, x):
    #   # first layer
    #   layer_1_road_rules = self.new_road_rules(left_lane = True, right_lane = False, up_lane = True, down_lane = True)
    #   # second layer
    #   layer_1_road_rules = self.new_road_rules(left_lane = True, right_lane = True, down_lane = True, up_lane = False)

    #   min_func = MinFuncMux()
    #   min_func.store(self.g_right_of_main, self.g_right_of_main(x, road_rules = self._road_rules))
    #   min_func.store(self.g_outside_rightband, self.g_outside_rightband(x, road_rules = self._road_rules))
    #   return min_func.get_min()      

    def g_right_combined(self, x):
      _min_func = MinFuncMux()
      _min_func.store(self.g_right_of_main, self.g_right_of_main(x).detach().numpy().flatten()[0])
      _min_func.store(self.g_outside_rightband, self.g_outside_rightband(x).detach().numpy().flatten()[0])
      return _min_func.get_min()

    def g_circle_intersection(self, x):
      _r = self._road_rules["width"]
      return _r**2 - (x[self._x_index, 0] - self._road_rules["x_max"] - _r) ** 2 - (x[self._y_index, 0] - self._road_rules["y_max"] - _r) ** 2

    def g_road_rules(self, x, **kwargs):
      # left_turn = self._road_logic["left_turn"]

      # for key in kwargs.keys():
      #   if key == "left_turn":
      #     left_turn = kwargs["left_turn"]

      # if not left_turn:
      self._max_func.store(self.g_left_of_main, self.g_left_of_main(x).detach().numpy().flatten()[0])
      _func_of_min_val, _min_val = self.g_right_combined(x)
      self._max_func.store(_func_of_min_val, _min_val)
      _func_of_max_val, _max_val = self._max_func.get_max()
      return _max_val, _func_of_max_val
      # else:
      #   return max(
      #     self.g_road_rules(car, False, left_lane = True, right_lane = False, up_lane = True, down_lane = True),
      #     self.g_road_rules(car, False, left_lane = True, right_lane = True, down_lane = True, up_lane = False),
      #     self.g_circle_intersection(car)
      #   )

    def __call__(self, x, k=0):
      # signed_distance = self._polyline.signed_distance_to(
      #     Point(x[self._x_index, 0], x[self._y_index, 0]))
      _max_val, _func_of_max_val = self.g_road_rules(x)
      return _max_val * torch.ones(1, 1, requires_grad=True).double(), _func_of_max_val

    def render(self, ax=None):
      """ Render this cost on the given axes. """
      xs = [pt.x for pt in self._polyline.points]
      ys = [pt.y for pt in self._polyline.points]
      ax.plot(xs, ys, "k", alpha=0.25)

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