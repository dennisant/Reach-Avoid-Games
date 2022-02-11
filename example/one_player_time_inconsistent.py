import os
import numpy as np

cmd = "python3 run.py                                       \
        --env_type goal_with_obs                            \
        --init_states 6.0 0.0 {} 0.0 8.0                    \
        --no_players 1                                      \
        --obstacles 9.0 25.0 4.5 20.0 35.0 3.0 6.5 46.0 3.0 \
        --goal 6.0 40.0 2.0                                 \
        --t_horizon 10                                      \
        --linesearch --linesearch_type trust_region         \
        ".format(np.pi / 2.01)

os.system(cmd)