import torch
import numpy as np
import math

from cost.cost import Cost
from utils.utils import MaxFuncMux

class CollisionPenalty(Cost):
    """
    Collision Penalty between two cars.
    """
    def __init__(self, g_params, name="collision_penalty"):
      self._position_indices = g_params["position_indices"]
      self._collision_r = g_params["collision_r"]
      self._car_params = g_params["car_params"]
      self._theta_indices = g_params["theta_indices"]
      self._player_id = g_params["player_id"]

      super(CollisionPenalty, self).__init__("car{}_".format(g_params["player_id"]+1)+name)

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