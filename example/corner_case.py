import os

cmd = "python3 run.py                                                                                   \
        --no_players 1                                                                                  \
        --env_type goal_with_obs                                                                        \
        --eps_control 0.1 --eps_state 0.3                                                               \
        --alpha_scaling trust_region                                                                    \
        --hallucinated                                                                                  \
        --obstacles                                                                                     \
            9.0 25.0 4.5                                                                                \
            20.0 35.0 3.0                                                                               \
            6.5 50.0 3.0                                                                                \
            -4.0 33.0 2.0                                                                               \
            -5.0 44.0 2.0                                                                               \
            2.5 50.0 2.0                                                                                \
        --goal                                                                                          \
            6.0 40.0 2.0                                                                                \
        --init_states -9.005303678348714 59.85495235317849 -1.085491720939467 0.0 5.315173749764264     \
        --t_horizon 9 --max_steps 150 --exp_name exp_time_inconsistent --time_consistency               \
    "

os.system(cmd)