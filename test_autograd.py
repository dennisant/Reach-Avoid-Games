import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from torch.autograd import grad
from car_5d import Car5Dv2

def sample_g(x):
    return torch.max(
        x[0, 0] + 3,
        2 - x[1, 0]
    )

def sample_random(x):
    return 2.0 - x[1, 0] ** 2

def sample_l(x):
    _road_rules = {
        "x_min": 2,
        "x_max": 9,
        "y_max": 17,
        "y_min": 10,
        "width": 3.5
    }

    value = torch.min(
        torch.max(
            20.0 - x[0, 0],
            torch.max(
                x[1, 0] - _road_rules["y_max"],
                _road_rules["y_min"] - x[1, 0]
            )
        ),
        torch.max(
            25.0 - x[1, 0],
            torch.max(
                x[0, 0] - _road_rules["x_max"],
                _road_rules["x_min"] - x[0, 0]
            )
        ),
    )

    return value

def sample_l_secondOrder(x):
    _road_rules = {
        "x_min": 2,
        "x_max": 9,
        "y_max": 17,
        "y_min": 10,
        "width": 3.5
    }

    value = torch.min(
        torch.max(
            abs(20.0 - x[0, 0]) * (20.0 - x[0, 0]),
            torch.max(
                abs(x[1, 0] - _road_rules["y_max"]) * (x[1, 0] - _road_rules["y_max"]),
                abs(_road_rules["y_min"] - x[1, 0]) * (_road_rules["y_min"] - x[1, 0])
            )
        ),
        torch.max(
            abs(25.0 - x[1, 0]) * (25.0 - x[1, 0]),
            torch.max(
                abs(x[0, 0] - _road_rules["x_max"]) * (x[0, 0] - _road_rules["x_max"]),
                abs(_road_rules["x_min"] - x[0, 0]) * (_road_rules["x_min"] - x[0, 0])
            )
        ),
    )

    return value

def sample_l_frontrear(x):
    _road_rules = {
        "x_min": 2,
        "x_max": 9,
        "y_max": 17,
        "y_min": 10,
        "width": 3.5
    }
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }
    car_rear, car_front = get_car_state(x)
    print("Car rear front: ", car_rear, car_front)
    _collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)
    return torch.max(
        torch.max(
            25.0 - car_rear[0],
            torch.max(
                car_rear[1] - _road_rules["y_max"] + _collision_r,
                _road_rules["y_min"] - car_rear[1] + _collision_r
            )
        ),
        torch.max(
            25.0 - car_front[0],
            torch.max(
                car_front[1] - _road_rules["y_max"] + _collision_r,
                _road_rules["y_min"] - car_front[1] + _collision_r
            )
        )
    )

def sample_l_singleOrder_unconstrained(x):
    return 25 - x[1, 0]

def sample_collision(x):
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }
    car1_x_index = 0
    car1_y_index = 1
    car2_x_index = 5
    car2_y_index = 6
    collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)
    return (4.0 * collision_r**2 - (x[car1_x_index, 0] - x[car2_x_index, 0]) ** 2 - (x[car1_y_index, 0] - x[car2_y_index, 0]) ** 2)

def get_car_state(x, car_x_index=0, car_y_index=1, theta_indices=2):
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }
    if type(x) is torch.Tensor:
        car_rear = x[car_x_index:2+car_x_index, 0]
        car_front = [
            x[car_x_index, 0] + car_params["wheelbase"]*torch.cos(x[theta_indices, 0]),
            x[car_y_index, 0] + car_params["wheelbase"]*torch.sin(x[theta_indices, 0])
        ]
    else:
        car_rear = np.array([x[car_x_index, 0], x[car_y_index, 0]])
        car_front = [
            x[car_x_index, 0] + car_params["wheelbase"]*math.cos(x[theta_indices, 0]),
            x[car_y_index, 0] + car_params["wheelbase"]*math.sin(x[theta_indices, 0])
        ]
    return car_rear, car_front

def calculate_car_front(x, car_x_index=0, car_y_index=1, theta_indices=2):
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }
    car_front = [
        x[car_x_index, 0] + car_params["wheelbase"]*torch.cos(x[theta_indices, 0]),
        x[car_y_index, 0] + car_params["wheelbase"]*torch.sin(x[theta_indices, 0])
    ]
    return car_front

def g_coll_fr(x):
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }
    
    _car1_rear, _car1_front = get_car_state(x, 0, 1, 2)
    _car2_rear, _car2_front = get_car_state(x, 5, 6, 7)
    collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

    if type(x) is torch.Tensor:
        return 2.0 * collision_r - torch.sqrt((_car1_front[0] - _car2_rear[0]) ** 2 + (_car1_front[1] - _car2_rear[1]) ** 2)
    else:
        return 2.0 * collision_r - math.sqrt((_car1_front[0] - _car2_rear[0]) ** 2 + (_car1_front[1] - _car2_rear[1]) ** 2)


def g_coll_rr(x):
    car_params = {
        "wheelbase": 2.413, 
        "length": 4.267,
        "width": 1.988
    }
    
    _car1_rear, _car1_front = get_car_state(x, 0, 1, 2)
    _car2_rear, _car2_front = get_car_state(x, 5, 6, 7)
    collision_r = math.sqrt((0.5 * (car_params["length"] - car_params["wheelbase"])) ** 2 + (0.5 * car_params["width"]) ** 2)

    if type(x) is torch.Tensor:
        return 2.0 * collision_r - torch.sqrt((_car1_rear[0] - _car2_rear[0]) ** 2 + (_car1_rear[1] - _car2_rear[1]) ** 2)
    else:
        return 2.0 * collision_r - math.sqrt((_car1_rear[0] - _car2_rear[0]) ** 2 + (_car1_rear[1] - _car2_rear[1]) ** 2)

######################
### DRAW ROAD AND CAR
######################
def draw_roads(road_rules, x_max, y_max):
    """
        x_max: max value of plot in x axis
        y_max: max value of plot in y axis
    """
    print(road_rules)
    # This function draw t-intersection based on road_rules
    x_center = road_rules["x_min"] + 0.5 * (road_rules["x_max"] - road_rules["x_min"])
    y_center = road_rules["y_min"] + 0.5 * (road_rules["y_max"] - road_rules["y_min"])
    plt.plot([road_rules["x_min"], road_rules["x_min"]], [0, x_max], c='k')
    plt.plot([road_rules["x_max"], road_rules["x_max"]], [0, road_rules["y_min"]], c='k')
    plt.plot([road_rules["x_max"], road_rules["x_max"]], [road_rules["y_max"], x_max], c='k')
    plt.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_min"], road_rules["y_min"]], c='k')
    plt.plot([road_rules["x_max"], y_max], [road_rules["y_min"], road_rules["y_min"]], c='k')
    plt.plot([road_rules["x_min"], road_rules["x_min"]], [road_rules["y_max"], road_rules["y_max"]], c='k')
    plt.plot([road_rules["x_max"], y_max], [road_rules["y_max"], road_rules["y_max"]], c='k')
    plt.plot([x_center, x_center], [0, x_max], "--", c = 'b')
    plt.plot([road_rules["x_max"], y_max], [y_center, y_center], "--", c = 'b')

def draw_car(cars):
    color = ["g", "b", "y"]
    for car in cars:
        plt.gca().set_aspect('equal')
        rear, front = car.get_poc()
        rotate_deg = car.state[2]/np.pi * 180
        length = car.length
        width = car.width
        wheelbase = car._l
        a = 0.5 * (length - wheelbase)

        plt.plot(rear[0], rear[1], color='r', marker='o', markersize=10)
        rec = plt.Rectangle(rear-np.array([a, 0.5*width]), width=length, height=width, color = color.pop(), alpha=0.8,
                                transform=Affine2D().rotate_deg_around(*(rear[0], rear[1]), rotate_deg)+plt.gca().transData)
        plt.gca().add_patch(rec)

car_params = {
    "wheelbase": 2.413, 
    "length": 4.267,
    "width": 1.988
}

road_rules = {
    "x_min": 2,
    "x_max": 9,
    "y_max": 17,
    "y_min": 10,
    "width": 3.5
}

car1 = Car5Dv2(T = 0.1, **car_params)
car2 = Car5Dv2(T = 0.1, **car_params)
fig = plt.figure(figsize = (10, 12.5))
draw_roads(road_rules, 40, 25)
car1.state = np.array([5, 20, np.pi/2.0, 0, 0])
car2.state = np.array([6, 21, -np.pi/4, 0, 0])
# draw_car([car1, car2])
draw_car([car1])

x = np.concatenate((car1.state, car2.state), axis=0).reshape(10, 1)
x_torch = torch.from_numpy(x).requires_grad_(True)
output = sample_l_frontrear(x_torch)

print(output)

grad_sample = torch.autograd.grad(outputs=output, inputs=x_torch, create_graph=True, allow_unused=True)[0]
print(grad_sample)

hess_x = np.zeros((len(x), len(x)))
if grad_sample is not None and grad_sample.grad_fn is not None:
    print("Grad sample is not None")
    for ii in range(len(x)):
        hess_row = torch.autograd.grad(
            grad_sample[ii, 0], x_torch, retain_graph=True)[0]
        hess_x[ii, :] = hess_row.detach().numpy().copy().T
# print(hess_x)

plt.show()