import os
import numpy as np

sample_run_1 = "python3 run.py                                      \
                --env_type goal_with_obs                            \
                --init_states 6.0 0.0 {} 0.0 8.0                    \
                --no_players 1                                      \
                --obstacles 9.0 25.0 4.5 20.0 35.0 3.0 6.5 50.0 3.0 \
                --goal 6.0 40.0 2.0                                 \
                --alpha_scaling trust_region                        \
                --t_horizon 12.0                                    \
                --eps_control 0.1 --eps_state 0.1                   \
                --hallucinated                                      \
                --plot --log                                        \
                ".format(np.pi/2.5)

sample_run_2 = "python3 run.py                                      \
                --env_type goal_with_obs                            \
                --init_states 15.0 17.0 {} 0.0 8.0                  \
                --no_players 1                                      \
                --obstacles                                         \
                        9.0 25.0 4.5                                \
                        20.0 35.0 3.0                               \
                        6.5 50.0 3.0                                \
                        -4.0 33.0 2.0                               \
                        -5.0 44.0 2.0                               \
                        2.5 50.0 2.0                                \
                --goal 6.0 40.0 2.0                                 \
                --alpha_scaling trust_region                        \
                --t_horizon 4.0                                     \
                --eps_control 0.1 --eps_state 0.1                   \
                --hallucinated                                      \
                --plot --log                                        \
                ".format(np.pi/3.0)

sample_run_3 = "python3 run.py                                      \
                --env_type goal_with_obs                            \
                --init_states 6.0 7.5 {} 0.0 8.0                    \
                --no_players 1                                      \
                --obstacles                                         \
                        9.0 25.0 4.5                                \
                        20.0 35.0 3.0                               \
                        6.5 50.0 3.0                                \
                        2.5 50.0 2.0                                \
                --goal 6.0 40.0 2.0                                 \
                --alpha_scaling trust_region                        \
                --t_horizon 12.0                                     \
                --eps_control 0.1 --eps_state 0.1                   \
                --hallucinated                                      \
                --plot --log                                        \
                ".format(np.pi/3.5)

sample_run_4 = "python3 run.py                                       \
                --env_type goal_with_obs                            \
                --init_states 6.0 0.0 {} 0.0 8.0                    \
                --no_players 1                                      \
                --obstacles                                         \
                        9.0 25.0 4.5                                \
                        20.0 35.0 3.0                               \
                        6.5 50.0 3.0                                \
                        2.5 50.0 2.0                                \
                        11.5 46.0 2.0                               \
                --goal 6.0 40.0 2.0                                 \
                --alpha_scaling trust_region                        \
                --trust_region_type ratio                           \
                --t_horizon 12.0                                    \
                --eps_control 0.1 --eps_state 0.1                   \
                --hallucinated                                      \
                --plot --log                                        \
                ".format(np.pi/4.0)

os.system(sample_run_4)