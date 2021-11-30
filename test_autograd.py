from typing_extensions import Required
import torch
import numpy as np

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

x = np.array([1., 20., 0., 0., 0., 1., 20., 0., 0., 0.]).reshape(10, 1)
x_torch = torch.from_numpy(x).requires_grad_(True)
output = sample_l_secondOrder(x_torch)

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
print(hess_x)