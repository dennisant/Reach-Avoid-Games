import os
import numpy as np

cmd = "python3 run.py                                       \
        --env_type t_intersection                           \
        --no_players 3                                      \
        --player_types car car ped                          \
        --init_states                                       \
            7.5 0.0 {} 0.0 15.0                              \
            3.75 40.0 {} 0.0 15.0                           \
            -2.0 30.0 0.0 2.0                               \
        --draw_roads                                        \
        --linesearch --linesearch_type trust_region         \
        --boundary_only                                     \
        --hallucinated                                      \
        ".format(np.pi/2.01, -np.pi/2.01)

# --draw_roads --draw_human --draw_cars

os.system(cmd)