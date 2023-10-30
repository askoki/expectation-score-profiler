import os
import re
import warnings

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from settings import FIGURES_DIR, REPORTS_DIR, USE_DUMMY, PSO_FIG_DIR, NELDER_MEAD_FIG_DIR
from src.visualization.constants import PS_REDUCED_COL_NAMES
from src.features.data_loaders import load_optimisation_data, load_df_data
from src.features.optimisation.processing import PlayerDataProcessor, OptInfo
from src.visualization.helpers.processing import get_best_run, convert_gd_to_step_fill_function

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
pd.options.mode.chained_assignment = None

data_holder = load_optimisation_data(is_dummy=USE_DUMMY)
data_df = load_df_data()
# player_games_df = data_df.groupby(['athlete', 'date']).agg({'date': 'first'}).groupby('athlete').agg(
#     {'date': 'count'}).reset_index().rename(columns={'date': 'num_games'})
# max_games_per_player = player_games_df.num_games.max()
data_df.loc[:, 'game_id'] = data_df.date.str.extract('(\d+)').astype(int)
max_games_per_player = data_df.game_id.max()

players_list = data_holder.get_players()
players_list = players_list
players_count = len(players_list)

plt.rc('ytick', labelsize=4)
plt.rc('xtick', labelsize=4)
plt.rc('axes', labelsize=5)

# GD chart is +1
n_rows = players_count + 1
fig, axs = plt.subplots(
    figsize=(0.6 * max_games_per_player, 0.6 * n_rows),
    nrows=n_rows,
    ncols=max_games_per_player,
    sharex='col',
    # gridspec_kw={'height_ratios': [0.8, 0.8] * max_games_per_player}
)
plot_matrix = np.zeros(axs.shape)
gd_holder = {}
for p_idx, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {p_idx + 1}/{players_count}')
    athlete_id = int(re.search('(\d+)', player_name).group(0))
    athlete_idx = athlete_id - 1

    axs[athlete_idx][0].text(
        -65, 5, s=player_name, fontsize=6, fontweight='bold', rotation=90
    )

    player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))
    best_run_df = pd.read_csv(os.path.join(REPORTS_DIR, f'{source_name}_best_it_df.csv'))

    save_path = os.path.join(FIGURES_DIR, folder_name, player_name)
    load_path = os.path.join(REPORTS_DIR, folder_name, player_name)
    all_df = pd.read_csv(os.path.join(load_path, 'all_runs_df.csv'))

    best_it_series = get_best_run(all_df)
    opt_info = OptInfo(best_it_series[PS_REDUCED_COL_NAMES])
    player_data_holder.calculate_energy_for_all_matches_reduced_approach(opt_info)

    num_games = len(player_data_holder.player_measurements['values'])
    for i, game in enumerate(player_data_holder.player_measurements['values']):
        game_id = int(re.search('(\d+)', game['game_label']).group(0))
        game_idx = game_id - 1

        real_values = game['real_values'].cumsum()
        calc_values = game['calculated_values'].cumsum()
        min_from_start = game['min_from_start']
        gd_arr = game['gd']

        norm_kilo_factor = 1000
        norm_real_values = real_values / norm_kilo_factor
        norm_calc_values = calc_values / norm_kilo_factor

        min_from_start = min_from_start + (90 - min_from_start[-1]) if min_from_start[-1] < 45 else min_from_start
        axs[athlete_idx][game_idx].fill_between(
            min_from_start, norm_real_values, y2=0, interpolate=True,
            color='steelblue', alpha=0.5
        )
        axs[athlete_idx][game_idx].plot(min_from_start, norm_calc_values, 'orange')

        axs[athlete_idx][game_idx].spines['top'].set_visible(False)
        axs[athlete_idx][game_idx].spines['right'].set_visible(False)

        plot_matrix[athlete_idx][game_idx] = 1
        try:
            current_gd_duration = gd_holder[game_idx]['min_from_start']
        except KeyError:
            gd_holder[game_idx] = {
                'min_from_start': min_from_start,
                'gd': gd_arr
            }

        if min_from_start.shape[0] > gd_holder[game_idx]['min_from_start'].shape[0]:
            gd_holder[game_idx] = {
                'min_from_start': min_from_start,
                'gd': gd_arr
            }
        plot_matrix[-1][game_idx] = 1

for row_idx in range(plot_matrix.shape[0]):
    for col_idx in range(plot_matrix.shape[1]):
        if row_idx == 0:
            axs[0][col_idx].text(35, 55, s=f'G {col_idx + 1}', fontsize=6)
        axs[row_idx][col_idx].set_xlim((0, 90))

        axs[-1][col_idx].set_ylim((-2.05, 2.05))
        axs[-1][col_idx].spines['top'].set_visible(False)
        x_step, y_step = convert_gd_to_step_fill_function(
            gd_holder[col_idx]['min_from_start'],
            gd_holder[col_idx]['gd']
        )
        axs[-1][col_idx].step(
            x_step,
            y_step, where='post', linewidth=1
        )
        axs[-1][col_idx].fill_between(
            x_step,
            y_step,
            where=(y_step > 0),
            interpolate=True, color='forestgreen', alpha=0.5
        )
        axs[-1][col_idx].fill_between(
            x_step,
            y_step,
            where=(y_step < 0),
            interpolate=True, color='tomato', alpha=0.5
        )

        # Only first column needs to have this information
        last_row_idx = plot_matrix.shape[0] - 1
        if col_idx == 0:
            position_horizontal = -80
            position_vertical = -30
            if row_idx == last_row_idx:
                axs[row_idx][0].set_ylabel('GD')
            else:
                axs[row_idx][0].set_ylabel('E kJ/kg')
        else:
            axs[row_idx][col_idx].set_yticks([])
            axs[row_idx][col_idx].spines['right'].set_visible(False)

        if row_idx < last_row_idx:
            axs[row_idx][col_idx].get_xaxis().set_visible(False)
            axs[row_idx][col_idx].set_ylim((0, 50))

        axs[row_idx][col_idx].spines['bottom'].set_visible(False)
        # disable unused
        if plot_matrix[row_idx][col_idx] == 0:
            axs[row_idx][col_idx].spines['right'].set_visible(False)
            axs[row_idx][col_idx].spines['top'].set_visible(False)
            axs[row_idx][col_idx].spines['bottom'].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0)
plt.savefig(
    os.path.join(FIGURES_DIR, folder_name, f'{source_name}_all_players_overview_plot.png'),
    dpi=300
)
