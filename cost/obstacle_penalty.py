import torch
import numpy as np
import matplotlib.pyplot as plt

from cost.cost import Cost
from resource.point import Point
import math

from utils.utils import MaxFuncMux

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

    def get_closest_obs(self, x):
        return np.argmax([self.g_single_obstacle_collision(x, obs_index=i) for i in range(len(self._obs))])

    def first_order(self, x, k=0):
        # figure out which obstacle is the closest, pass back index of obs in self._obs
        obs_index = self.get_closest_obs(x)
        # use that information for first order
        A = self._obs[obs_index].x - x[self._x_index, 0]
        B = self._obs[obs_index].y - x[self._y_index, 0]
        C = A ** 2 + B ** 2
        return np.array([
            A / C ** 0.5, 
            B / C ** 0.5,
            0.0,
            0.0,
            0.0
        ])

    def second_order(self, x, k=0):
        # figure out which obstacle is the closest
        obs_index = self.get_closest_obs(x)
        # use that information for second order
        A = self._obs[obs_index].x - x[self._x_index, 0]
        B = self._obs[obs_index].y - x[self._y_index, 0]
        C = A ** 2 + B ** 2
        Dxx = -B**2 / C ** 1.5
        Dyy = -A**2 / C ** 1.5
        Dxy = Dyx = (-A * -B) / C ** 1.5
        return np.array([
            [Dxx, Dxy, 0, 0, 0],
            [Dyx, Dyy, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

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
                color="k", fill=True, alpha=0.75)
            ax.add_artist(circle)