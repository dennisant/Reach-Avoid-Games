import torch
import numpy as np
import matplotlib.pyplot as plt

from cost.cost import Cost
from resource.point import Point
import math

from utils import MaxFuncMux

class ObstacleDistCost(Cost):
    def __init__(self, g_params, name=""):
        """
        Initialize with dimension to add cost to and a max distance beyond
        which we impose no additional cost.
        :param position_indices: indices of input corresponding to (x, y)
        :type position_indices: (uint, uint)
        :param point: center of the obstacle from which to compute distance
        :type point: Point
        :param max_distance: maximum value of distance to penalize
        :type threshold: float
        """
        self._x_index, self._y_index = g_params["position_indices"][g_params["player_id"]]
        self._player_id = g_params["player_id"]
        self._obs = []

        for obs in g_params["obstacles"]:
            self._obs.append(Point(obs[0], obs[1]))
        
        self._obs_radii = g_params["obstacle_radii"]
        self._collision_r = g_params["collision_r"]
        super(ObstacleDistCost, self).__init__(name)

    def g_single_obstacle_collision(self, x, k=0, obs_index=0):
        dx = x[self._x_index, 0] - self._obs[obs_index].x
        dy = x[self._y_index, 0] - self._obs[obs_index].y
        if type(x) is torch.Tensor:            
            relative_distance = torch.sqrt(dx*dx + dy*dy)
        else:
            relative_distance = math.sqrt(dx*dx + dy*dy)
        return self._obs_radii[obs_index] - relative_distance + self._collision_r

    def g_obstacle_collision(self, x, k=0):
        value = self.g_single_obstacle_collision(x, obs_index=0)
        for i in range(len(self._obs)-1, 0, -1):
            if type(x) is torch.Tensor:
                value = torch.max(
                    self.g_single_obstacle_collision(x, obs_index=i),
                    value
                )
            else:
                value = max(
                    self.g_single_obstacle_collision(x, obs_index=i),
                    value
                )
        return value

    def __call__(self, x, k=0):
        _max_func = MaxFuncMux()
        _max_func.store(self.g_obstacle_collision, self.g_obstacle_collision(x))
        _func_of_max_val, _max_val = _max_func.get_max()
        return _max_val, _func_of_max_val

    def render(self, ax=None):
        """ Render this obstacle on the given axes. """
        for i in range(len(self._obs)):
            circle = plt.Circle(
                (self._obs[i].x, self._obs[i].y), self._obs_radii[i],
                color="r", fill=True, alpha=0.75)
            ax.add_artist(circle)