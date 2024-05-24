---
permalink: /docs/additional_scripts
layout: page
title: Additional relevant scripts and notebooks
---
### Plotting
All code used for preparing the figures used in the paper is in `scripts/plotting`. 
Scripts are expected to be run with ipython. Alternatively, they should be moved
to the root of the repo to access the required files. 

- `plot_aggregation.py` => Fig. (5c).
- `plot_containment_reg_bar.py` => Fig. (8). 
- `plot_containment_ranking.py` => Fig. (2).
- `plot_performance_data_lakes.py` => Fig. (7).
- `plot_starmie_results.py` => Fig. (4).
- `plot_time_breakdown.py` => Fig. (9).
- `plot_tradeoffs.py` => Fig. (10).
- `plot_topk_fulljoin.py` => Fig. (6).

### Notebooks
Notebooks are in `notebooks`. 

- `Stats on data lakes.ipynb` measures data lake stats that are then reported in Tab. (3). 
- `Run cleanup.iypnb` processes and puts together all the different experiments in singular files. These are the files that are used to prepare the plots. 
