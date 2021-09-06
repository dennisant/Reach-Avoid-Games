# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

from cost import Cost
from point import Point

class ObstacleCost(Cost):
    def __init__(self, position_indices, point, max_distance, name=""):
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
        self._x_index, self._y_index = position_indices
        self._point = point
        self._max_distance = max_distance
        super(ObstacleCost, self).__init__(name)

    def __call__(self, x, k=0):
        """
        Evaluate this cost function on the given input state.
        NOTE: `x` should be a column vector.
        :param x: concatenated state of the two systems
        :type x: torch.Tensor
        :return: scalar value of cost
        :rtype: torch.Tensor
        """
        # Compute relative distance.
        dx = x[self._x_index, 0] - self._point.x
        dy = x[self._y_index, 0] - self._point.y
        relative_distance = torch.sqrt(dx*dx + dy*dy)

        return relative_distance - self._max_distance

    def render(self, ax=None):
        """ Render this obstacle on the given axes. """
        circle = plt.Circle(
            (self._point.x, self._point.y), self._max_distance,
            color="r", fill=True, alpha=0.75)
        ax.add_artist(circle)
        ax.text(self._point.x - 1.25, self._point.y - 1.25, "obs", fontsize=8)