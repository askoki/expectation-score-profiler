expectation-score-profiler
==============================

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    │
    │
    ├── reports            <- Generated analysis as csv, png, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Helper scripts for optimisation and processing
    │   │
    │   ├── models         <- Scripts to start optimisation
    │   │   ├── nelder-mead
    │   │   │   └── optimize.py
    │   │   ├── pso
    │   │   │   └── optimize.py
    │   │   ├── collect_results.py <- generate .csv files in reports folder
    │   │   ├── compare_with_baseline.py <- compare pso and nm results with B1 and B2
    │   │   └── constants.py
    │   │  
    │   ├── reports  <- Scripts to create manuscript tables
    │   │   └── generate_latex_table_opt_results.py
    │   │  
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── create_example_opt_plot.py
    │       ├── generate_all_players_data_matrix.py
    │       ├── generate_all_players_overview_plots.py
    │       ├── generate_best_run_var_convergence.py
    │       ├── generate_cost_fun_through_eval.py
    │       ├── generate_player_data_matrix.py
    │       ├── generate_player_energy_distributions.py
    │       ├── generate_player_gd_minutes.py
    │       ├── generate_results_radars.py
    │       ├── generate_single_player_overview_plot.py
    │       └── generate_transition_matrices.py
    │
    └── settings.py            <- Project related configuration and settings.

--------
