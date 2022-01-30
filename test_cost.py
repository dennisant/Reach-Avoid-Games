from test.test_manual_derivative import ProximityCostTest
import numpy as np
import math
from resource.car_5d import Car5Dv2

car_params = {
    "wheelbase": 2.413, 
    "length": 4.267,
    "width": 1.988
}

collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

g_params = {
    "car": {
        "position_indices": [(0,1)],
        "player_id": 0, 
        "collision_r": collision_r,
        "car_params": car_params,
        "theta_indices": [2],
        "phi_index": 3, 
        "vel_index": 4,
        "obstacles": [
            (9.0, 25.0),
            (20.0, 35.0),
            (6.5, 46.0)
        ],
        "obstacle_radii": [
            4.5, 3.0, 3.0
        ]
    }
}

l_params = {
    "car": {
        "goals": [
            (6.0, 40.0)
        ],
        "goal_radii": [
            2.0
        ]
    }
}

config = {
    "g_params": g_params,
    "l_params": l_params
}

if __name__ == "__main__":
    print("TEST COST: ProximityCost")
    car1 = Car5Dv2(T = 0.1, **car_params)
    # car2 = Car5Dv2(T = 0.1, **car_params)
    car1.state = np.array([5, 20, np.pi/2.0, 0, 0])
    # car2.state = np.array([6, 21, -np.pi/4, 0, 0])

    # x = np.concatenate((car1.state, car2.state), axis=0).reshape(10, 1)
    x = car1.state.reshape(5, 1)
    test = ProximityCostTest(config)

    print("First order: {}".format(test.compare_first_order(x)))
    print("Second order: {}".format(test.compare_second_order(x)))