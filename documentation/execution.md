---
permalink: /docs/execution
layout: page
---
**NOTE:** We recommend to use the smaller `binary_update` data lake and its corresponding configurations to set up the data structures and debug potential issues, as all preparation steps are significantly faster than with larger data lakes. 

The configurations used to run the experiments in the paper are available in directory `config/evaluation`. 

The experiment configurations that tested default parameters are stored in `config/evaluation/general`; experiment configurations 
testing aggregation are in `config/evaluation/aggregation`; additional experiments that test specific parameters and scenarios are in `config/evaluation/other`.

To run experiments with `binary_update`:
```sh
python main.py config/evaluation/general/config-binary.toml
```
