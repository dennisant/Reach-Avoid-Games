import os
import numpy as np

cmd = "python3 run.py                                       \
        --env_type t_intersection                           \
        --no_players 3                                      \
        --player_types car car ped                          \
        --init_states                                       \
            7.5 0.0 {} 0.0 5.0                           \
            3.75 40.0 {} 0.0 10.0                       \
            -2.0 30.0 0.0 2.0                               \
        --time_consistency                                  \
        --draw_roads --draw_human --draw_cars               \
        ".format(np.pi/2.01, -np.pi/2.01)

os.system(cmd)