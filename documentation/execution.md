---
permalink: /docs/execution
layout: page
prev_page: /docs/preparation
---
**NOTE:** We recommend to use the smaller `binary_update` data lake and its corresponding configurations to set up the data structures and debug potential issues, as all preparation steps are significantly faster than with larger data lakes, and it is less likely to run into runtime or memory issues. 

The configurations used to run the experiments in the paper are available in directory `config/evaluation`. 

The experiment configurations that tested default parameters are stored in `config/evaluation/general`; experiment configurations 
testing aggregation are in `config/evaluation/aggregation`; additional experiments that test specific parameters and scenarios are in `config/evaluation/other`.

For clarity, by `experiment` we refer to a single call of the `main.py` script, during which the configuration file is read and a grid of parameters is built. Each combination of parameters in the grid is a `run`; an experiment consists of at least one run, and usually multiple. 

Be aware that the experiment configuration is parsed and the parameter grid is built greedily by creating all possible configurations of parameters. This means that if some configurations are not available (e.g., Starmie on Open Data US), an exception will be raised and the experiment will fail.

The `main.py` script is the entry point for the pipeline. It is possible to run the code using a configuration file such as those provided above, or it is possible to recover from a failed experiment by providing the path to the run that should be recovered. 

In the latter case, the `main.py` script will prepare a new experiment to execute all the missing configurations. The user will then have to combine the result of the two experiments. 

**NOTE ON MAX THREADS:** We fix the number of polars threads to 32 for reproducibility reasons. Depending on the user 
scenario, this value might have to be modified. This can be done by editing the value in the line:
```py
os.environ["POLARS_MAX_THREADS"] = "32"
```

To run experiments with a default `binary_update` configuration:
```sh
python main.py --input_path config/evaluation/general/config-binary.toml
```

To recover from a failed run with path `results/logs/0111-yoiea59a`:
```sh
python main.py --recovery_path results/logs/0111-yoiea59a
```

By adding the `-a` or `--archive` argument, the folder of the current run will be compressed in tar and added to the 
folder `results/archives`. 

## Parameter validation
Prior to executing the pipeline, `main.py` will validate all the provided configurations and check that all parameters are correct, and that all required data is available to the script. This ensures that the experiment will not fail halfway through because a specific configuration is missing something. 

An exception will be raised if any configuration is found to be incorrect. 

## Logging the run results
An extensive logging architecture was set up to track all configurations, parameters and metrics of interest that we used for the paper. 

For each run configuration we track:
- total runtime
- time spent in different sections of the code (join, train, prepare)
- memory utilization throughout the execution
- prediction performance (R2 or AUC depending on the task)

Each time  `main.py` is run, a new `scenario` will be created with a unique ID 
that tracks the current experiment number (stored in `results/scenario_id`). For each scenario, the script creates a new folder with the same name as the scenario ID 
(e.g., `0111-yoiea59a`). 

The folder contains the subfolders `json` and `run_logs`, a file
named `missing_runs.pickle` that contains any missing configurations if the experiment failed, and a cfg file that copies the configuration used to prepare the current experiment. 

Subfolder `json` contains a json file for each parameter configuration, which contains the parameters for each specific run, as well as all the associated metrics. Subfolder `run_logs` contains a `.log` file which reports the prediction results and 
for the given parameters for each crossvalidation fold.   

This architecture allows to keep track of the parameters used in the experiments as well as possible. 

