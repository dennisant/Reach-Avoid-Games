# How to run
Run ```run.py``` directly with flags to change the env configs
```
python3 run.py [flags]
```
Example:
```
python3 run.py --no_players 1 --env_type goal_with_obs --init_states 6.0 0.0 1.563 0.0 8.0
```
Or run our samples listed in *./example*:
```
python3 example/<name of example>
```
For example:
```
python3 example/one_player_time_consistent.py
```

# Batch run
There is a ```run_batch.py``` file to help automatically generate randomized initialization data for multiple runs for one-player case. Change the initialization range in the script before running to match your targeted test cases. You can either choose to run only **time_consistent**, **time_inconsistent** or **both**. If **both** is chosen, the test cases across time consistetn and inconsistent will be the same. Each experiment in the batch will have its own log file and figures, there will also be a common batch log to record all commands used to run the experiments in the batch.
**Note**: If you do not want to either have logs or plots, or methods of running for the batch run, change information in the **base_flag** in the script.
```
base_flag = "   python3 run.py                      \
                --no_players 1                      \
                --env_type goal_with_obs            \
                --eps_control 0.1 --eps_state 0.1   \
                --linesearch                        \
                --alpha_scaling trust_region        \
                --batch_run                         \
                --plot                              \
                --log                               \
                --hallucinated                      \
            "
```

Once finish running, you can analyze the data in the batch by either running ```analyze.ipynb```. Each batch can be imported as a ```Batch``` object with certain functions. You can also quickly check on the convergence rate of the batch run and see the resulting plots by running ```evaluate_batch.py```.
```
python3 evaluate_batch.py --loadpath result/batch-2022-02-19/ --exp_suffix exp_time_consistent
```