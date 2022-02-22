import torch
import numpy as np
import matplotlib.pyplot as plt
from cost.cost import Cost

class PedestrianProximityToBlockCost(Cost):
  """
  Proximity cost of pedestrian, used as l function to reach target.
  """
  def __init__(self, g_params, name=""):
    self._x_index, self._y_index = g_params["position_indices"][g_params["player_id"]]
    self._goal_x, self._goal_y = g_params["goals"]
    self._player_id = g_params["player_id"]
    self._road_rules = g_params["road_rules"]
    self._road_logic = self.get_road_logic_dict(g_params["road_logic"])
    self._road_rules = self.new_road_rules()
    self._theta_indices = g_params["theta_indices"]

    super(PedestrianProximityToBlockCost, self).__init__(name)

  def get_road_logic_dict(self, road_logic):
    return {
      "left_lane": road_logic[0] == 1, 
      "right_lane": road_logic[1] == 1, 
      "up_lane": road_logic[2] == 1, 
      "down_lane": road_logic[3] == 1, 
      "left_turn": road_logic[4] == 1
    }

  def __call__(self, x, k=0):
    if type(x) is torch.Tensor:
      max_func = torch.max
    else:
      max_func = max

    value = max_func(
      self._goal_x - x[self._x_index, 0],
      max_func(
        x[self._y_index, 0] - self._road_rules["y_max"],
        self._road_rules["y_min"] - x[self._y_index, 0]
      )
    )      

    return value

  def render(self, ax=None, contour = False):
    """ Render this obstacle on the given axes. """
    # ax.plot([self._road_rules["x_min"], self._road_rules["x_max"]], [self._goal_y, self._goal_y], c = 'r', linewidth = 10, alpha = 0.2)
    # ax.plot([self._goal_x, self._goal_x], [self._road_rules["y_min"], self._road_rules["y_max"]], c = 'b', linewidth = 10, alpha = 0.4)

    goal = plt.Rectangle(
        [self._goal_x, self._road_rules["y_min"]], width = 10, height = self._road_rules["y_max"] - self._road_rules["y_min"], color = "b", lw = 0, alpha = 0.4)
    ax.add_patch(goal)
    # ax.text(self._goal_x - 1, self._road_rules["y_min"] - 0.5, "ped_goal", fontsize=10)

    if contour:
      self.target_contour(ax)
  
  def target_contour(self, ax=None):
    x_range = np.arange(0, 25, step = 0.1)
    y_range = np.arange(0, 40, step = 0.1)
    zz = np.array([[0]*250]*400)
    for x in x_range:
        for y in y_range:
            xs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, y, 0, 0]).reshape(14, 1)
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