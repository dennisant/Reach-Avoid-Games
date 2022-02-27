################################################################################
#
# Proximity cost, derived from Cost base class. Implements a cost function that
# depends only on state and penalizes -min(distance, max_distance)^2.
#
################################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt

from cost.cost import Cost
import math

class ProximityToBlockCost(Cost):
  """
  Proximity cost, with goal is an infinitely expanding block instead of a circle.
  """
  def __init__(self, g_params, name=""):
    self._x_index, self._y_index = g_params["position_indices"][g_params["player_id"]]
    self._goal_x, self._goal_y = g_params["goals"]
    self._player_id = g_params["player_id"]
    self._road_rules = g_params["road_rules"]
    self._road_logic = self.get_road_logic_dict(g_params["road_logic"])
    self._road_rules = self.new_road_rules()
    self._collision_r = g_params["collision_r"]
    # self._collision_r = 0
    self._car_params = g_params["car_params"]
    self._theta_indices = g_params["theta_indices"]
    self.use_rear_front = True

    super(ProximityToBlockCost, self).__init__(name)

  def get_road_logic_dict(self, road_logic):
    return {
      "left_lane": road_logic[0] == 1, 
      "right_lane": road_logic[1] == 1, 
      "up_lane": road_logic[2] == 1, 
      "down_lane": road_logic[3] == 1, 
      "left_turn": road_logic[4] == 1
    }

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

  def target_contour(self, ax=None):
      x_range = np.arange(0, 25, step = 0.1)
      y_range = np.arange(0, 40, step = 0.1)
      zz = np.array([[0]*250]*400)
      for x in x_range:
          for y in y_range:
              xs = np.array([x, y, np.pi * 0.5, 0, 0, x, y, -np.pi * 0.5, 0, 0]).reshape(10, 1)
              zz[int(y*10)][int(x*10)] = self(xs)
      # contour = ax.contourf(x_range, y_range, zz, cmap = "YlGn", alpha = 0.5, levels = np.arange(-10, 20, step=1))
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

class ProximityToUpBlockCost(ProximityToBlockCost):
  """
  Proximity cost, with goal is an infinitely expanding block instead of a circle.
  This variation is for up block, for car that goes straight from down to upward
  """
  def __init__(self, g_params, name=""):
    self.use_rear_front = True
    super().__init__(g_params, name)

  def __call__(self, x, k=0):
    car_rear, car_front = self.get_car_state(x, self._player_id)
    
    if type(x) is torch.Tensor:
      max_func = torch.max
    else:
      max_func = max
    
    if not self.use_rear_front:
      value = max_func(
        self._goal_y - x[self._y_index, 0],
        max_func(
          x[self._x_index, 0] - self._road_rules["x_max"] + self._collision_r,
          self._road_rules["x_min"] - x[self._x_index, 0] + self._collision_r
        )
      )
    else:
      value = max_func(
        max_func(
          self._goal_y - car_rear[1],
          max_func(
            car_rear[0] - self._road_rules["x_max"] + self._collision_r,
            self._road_rules["x_min"] - car_rear[0] + self._collision_r
          )
        ),
        max_func(
          self._goal_y - car_front[1],
          max_func(
            car_front[0] - self._road_rules["x_max"] + self._collision_r,
            self._road_rules["x_min"] - car_front[0] + self._collision_r
          )
        )
      )
    
    return value

  def render(self, ax=None, contour = False, player=0):
    """ Render this obstacle on the given axes. """

    goal = plt.Rectangle(
      [self._road_rules["x_min"], self._goal_y], width = self._road_rules["x_max"] - self._road_rules["x_min"], height = 10, color = "green", lw = 0, alpha = 0.4)
    ax.add_patch(goal)
    
    if contour and self._player_id == player:
      self.target_contour(ax)

class ProximityToLeftBlockCost(ProximityToBlockCost):
  """
  Proximity cost, with goal is an infinitely expanding block instead of a circle.
  This variation is for left block, for car that go from the top and turn left to t-intersection
  """
  def __init__(self, g_params, name=""):
    self.use_rear_front = True
    super().__init__(g_params, name)

  def __call__(self, x, k=0):
    car_rear, car_front = self.get_car_state(x, self._player_id)
    
    if type(x) is torch.Tensor:
      max_func = torch.max
    else:
      max_func = max

    if not self.use_rear_front:
      value = max_func(
        self._goal_x - x[self._x_index, 0],
        max_func(
          x[self._y_index, 0] - self._road_rules["y_max"] + self._collision_r,
          self._road_rules["y_min"] - x[self._y_index, 0] + self._collision_r
        )
      )
    else:
      value = max_func(
        max_func(
          self._goal_x - car_rear[0],
          max_func(
            car_rear[1] - self._road_rules["y_max"] + self._collision_r,
            self._road_rules["y_min"] - car_rear[1] + self._collision_r
          )
        ),
        max_func(
          self._goal_x - car_front[0],
          max_func(
            car_front[1] - self._road_rules["y_max"] + self._collision_r,
            self._road_rules["y_min"] - car_front[1] + self._collision_r
          )
        )
      )
    
    return value

  def render(self, ax=None, contour = False, player=0):
    """ Render this obstacle on the given axes. """
    goal = plt.Rectangle(
      [self._goal_x, self._road_rules["y_min"]], width = 10, height = self._road_rules["y_max"] - self._road_rules["y_min"], color = "r", lw = 0, alpha = 0.4)
    ax.add_patch(goal)

    if contour and self._player_id == player:
      self.target_contour(ax)

class ProximityToRightBlockCost(ProximityToBlockCost):
  """
  Proximity cost, with goal is an infinitely expanding block instead of a circle.
  This variation is for right block, for car that goes from down the turn right
  """
  def __init__(self, g_params, name=""):
    self.use_rear_front = True
    super().__init__(g_params, name)

  def __call__(self, x, k=0):
    car_rear, car_front = self.get_car_state(x, self._player_id)

    if type(x) is torch.Tensor:
      max_func = torch.max
    else:
      max_func = max
      
    if self.use_rear_front:
      value = max_func(
        self._goal_x - x[self._x_index, 0],
        max_func(
          x[self._y_index, 0] - self._road_rules["y_max"] + self._collision_r,
          self._road_rules["y_min"] - x[self._y_index, 0] + self._collision_r
        )
      )
    else:
      value = max_func(
        max_func(
          self._goal_x - car_rear[0],
          max_func(
            car_rear[1] - self._road_rules["y_max"] + self._collision_r,
            self._road_rules["y_min"] - car_rear[1] + self._collision_r
          )
        ),
        max_func(
          self._goal_x - car_front[0],
          max_func(
            car_front[1] - self._road_rules["y_max"] + self._collision_r,
            self._road_rules["y_min"] - car_front[1] + self._collision_r
          )
        )
      )
    
    return value

  def render(self, ax=None, contour = False, player=0):
    """ Render this obstacle on the given axes. """
    goal = plt.Rectangle(
      [self._goal_x, self._road_rules["y_min"]], width = 10, height = self._road_rules["y_max"] - self._road_rules["y_min"], color = "r", lw = 0, alpha = 0.4)
    ax.add_patch(goal)

    if contour and self._player_id == player:
      self.target_contour(ax)

class ProximityToDownBlockCost(ProximityToBlockCost):
  """
  Proximity cost, with goal is an infinitely expanding block instead of a circle.
  """
  def __init__(self, g_params, name=""):
    self.use_rear_front = True
    super().__init__(g_params, name)

  def __call__(self, x, k=0):
    car_rear, car_front = self.get_car_state(x, self._player_id)
    
    if type(x) is torch.Tensor:
      max_func = torch.max
    else:
      max_func = max
    
    if self.use_rear_front:
      value = max_func(
        x[self._y_index, 0] - self._goal_y,
        max_func(
          x[self._x_index, 0] - self._road_rules["x_max"],
          self._road_rules["x_min"] - x[self._x_index, 0]
        )
      )
    else:
      value = max_func(
        max_func(
          car_rear[1] - self._goal_y,
          max_func(
            car_rear[0] - self._road_rules["x_max"],
            self._road_rules["x_min"] - car_rear[0]
          )
        ),
        max_func(
          car_front[1] - self._goal_y,
          max_func(
            car_front[0] - self._road_rules["x_max"],
            self._road_rules["x_min"] - car_front[0]
          )
        )
      )
    return value

  def render(self, ax=None, contour = False, player=0):
    """ Render this obstacle on the given axes. """
    goal = plt.Rectangle(
      [self._road_rules["x_min"], self._goal_y - 10], width = self._road_rules["x_max"] - self._road_rules["x_min"], height = 10, color = "red", lw = 0, alpha = 0.4)
    ax.add_patch(goal)

    if contour and self._player_id == player:
      self.target_contour(ax)