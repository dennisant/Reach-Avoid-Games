import torch
import numpy as np
import math

from cost import Cost
from utils import MaxFuncMux

class ManeuverPenalty(Cost):
    def __init__(self, g_params, name="maneuver_penalty"):
      self._position_indices = g_params["position_indices"]
      self._collision_r = g_params["collision_r"]
      self._car_params = g_params["car_params"]
      self._phi_index = g_params["phi_index"]
      self._vel_index = g_params["vel_index"]
      self._max_vel = 25.0
      self._max_phi = np.pi / 4.0

      super(ManeuverPenalty, self).__init__("car{}_".format(g_params["player_id"]+1)+name)
    
    def g_speeding(self, x, k=0, **kwargs):
        return x[self._vel_index, 0] - self._max_vel

    def g_backward(self, x, k = 0, **kwargs):
        return -x[self._vel_index, 0]

    def g_steering(self, x, k=0, **kwargs):
        return x[self._phi_index, 0] ** 2 - self._max_phi ** 2

    def g_maneuver(self, x, **kwargs):
        _max_func = MaxFuncMux()
        _max_func.store(self.g_speeding, self.g_speeding(x))
        _max_func.store(self.g_backward, self.g_backward(x))
        _max_func.store(self.g_steering, self.g_steering(x))
        func_of_max_val, max_val = _max_func.get_max()
        return max_val, func_of_max_val

    def __call__(self, x, k=0):
      max_val, func_of_max_val = self.g_maneuver(x)
      return max_val, func_of_max_val