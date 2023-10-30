import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from settings import FIGURES_DIR, REPORTS_DIR, ETA_SIGN, PSO_FIG_DIR, NELDER_MEAD_FIG_DIR, USE_DUMMY
from src.features.data_loaders import load_optimisation_data
from src.features.file_helpers import create_dir
from src.features.optimisation.processing import PlayerDataProcessor, OptInfo
from src.visualization.constants import PS_REDUCED_COL_NAMES
from src.visualization.helpers.matplotlib_style import load_plt_style
from src.visualization.helpers.processing import get_best_run
from src.models.constants import bounds_power

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

    best_it_df = pd.read_csv(os.path.join(REPORTS_DIR, f'{source_name}_best_it_df.csv'))
    save_path = os.path.join(FIGURES_DIR, folder_name, player_name)
    load_path = os.path.join(REPORTS_DIR, folder_name, player_name)
    all_df = pd.read_csv(os.path.join(load_path, 'all_runs_df.csv'))

    create_dir(save_path)

    best_it_series = get_best_run(all_df)
    ps = OptInfo(best_it_series[PS_REDUCED_COL_NAMES])

    load_plt_style()
    fig = plt.figure(figsize=(10, 6))
    plt.ylabel('Variable values')
    plt.xlabel('Iteration')

    best_df = pd.read_csv(
        os.path.join(load_path, f'run_{best_it_series.iteration_count}_opt_variable_steps.csv')
    )
    # to make columns equal names with PowerDict
    best_df.columns = best_df.columns.str.replace('a', '')
    best_df.columns = best_df.columns.str.replace('v', '')
    best_df.columns = best_df.columns.str.replace('non', 'nf')
    best_df.columns = best_df.columns.str.replace('d', '')
    best_df.columns = best_df.columns.str.replace('l', '')
    best_df.columns = best_df.columns.str.replace('s', '')

    # normalize values to fit on the same scale
    for col in best_df.columns.values[:-1]:
        best_df.loc[:, col] = (best_df.loc[:, col] - bounds_power[0]) / (
                bounds_power[1] - bounds_power[0])
    # tmi instead of stamina because of chr removal earlier
    best_df = best_df.rename(columns={'tmin': ETA_SIGN})

    best_df.columns = best_df.columns.str.replace('_', ',')
    parameters = best_df.columns
    param_names = []
    for param in parameters:
        label_param = param
        if param != ETA_SIGN:
            label_param = f'$P_{{{param}}}$'
        param_names.append(label_param)
        plt.plot(best_df.index.values, best_df[param].values, label=label_param)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    fig.legend(
        loc='upper center', labels=param_names,
        bbox_to_anchor=(0.5, 0.45), ncol=5, fancybox=True,
        fontsize=10
    )
    fig.savefig(os.path.join(save_path, 'best_run_variables.png'), dpi=300)
    plt.close()
