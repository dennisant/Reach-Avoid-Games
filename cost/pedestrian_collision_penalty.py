import torch
import numpy as np
import math

from cost.cost import Cost
from utils.utils import MaxFuncMux

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