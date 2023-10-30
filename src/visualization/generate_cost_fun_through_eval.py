import os
import warnings
from argparse import ArgumentParser

from settings import FIGURES_DIR, REPORTS_DIR, PSO_FIG_DIR, NELDER_MEAD_FIG_DIR, USE_DUMMY
from src.features.data_loaders import load_optimisation_data
from src.features.optimisation.processing import PlayerDataProcessor
from src.models.constants import NUM_RUNS
from src.visualization.helpers.matplotlib_style import load_plt_style

import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser(description='Please provide data source info')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--pso', dest='is_pso', action='store_true')
group.add_argument('--nm', dest='is_pso', action='store_false')

is_pso = parser.parse_args().is_pso
source_path = PSO_FIG_DIR if is_pso else NELDER_MEAD_FIG_DIR
source_name = 'pso' if is_pso else 'nm'
folder_name = 'pso' if is_pso else 'nelder-mead'

# keep clean terminal
warnings.simplefilter(action='ignore', category=FutureWarning)
data_holder = load_optimisation_data(is_dummy=USE_DUMMY)
pd.options.mode.chained_assignment = None

players_list = data_holder.get_players()
players_count = len(players_list)

for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))

    best_run_df = pd.read_csv(os.path.join(REPORTS_DIR, f'{source_name}_best_run_df.csv'))
    save_path = os.path.join(FIGURES_DIR, folder_name, player_name)
    load_path = os.path.join(REPORTS_DIR, folder_name, player_name)
    all_df = pd.read_csv(os.path.join(load_path, 'all_runs_df.csv'))

    load_plt_style()
    fig = plt.figure(figsize=(10, 6))
    plt.ylabel('Cost function')
    plt.xlabel('Evaluations')
    for i in range(NUM_RUNS):
        eval_df = pd.read_csv(os.path.join(load_path, f'run_{i}_cost_fun_evaluations.csv'))
        plt.plot(eval_df.num_eval.values, eval_df.cost_fun.values, label=f'Run {i + 1}')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'cost_function_through_evaluations.png'), dpi=300)
    plt.close()