import os

import numpy as np
import pandas as pd
from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from settings import PSO_FIG_DIR, NELDER_MEAD_FIG_DIR, REPORTS_DIR, FIGURES_DIR, USE_DUMMY, INTERIM_DATA_DIR
from src.features.data_loaders import load_optimisation_data
from src.features.file_helpers import create_dir
from src.features.optimisation.processing import PlayerDataProcessor, OptInfo
from src.visualization.constants import PS_REDUCED_COL_NAMES, GD_STATES
from src.visualization.helpers.matplotlib_style import load_plt_style
from src.visualization.helpers.processing import add_labels_to_cm

parser = ArgumentParser(description='Please provide data source info')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--pso', dest='is_pso', action='store_true')
group.add_argument('--nm', dest='is_pso', action='store_false')

is_pso = parser.parse_args().is_pso
source_path = PSO_FIG_DIR if is_pso else NELDER_MEAD_FIG_DIR
save_name = 'pso' if is_pso else 'nm'

best_runs_df = pd.read_csv(os.path.join(REPORTS_DIR, f'{save_name}_best_run_df.csv'))
data_holder = load_optimisation_data(is_dummy=USE_DUMMY)
players_list = data_holder.get_players()
players_count = len(players_list)

comp_df = pd.DataFrame()
load_plt_style(is_default=True)
state_labels = ['GD-2', 'GD-1', 'GD0', 'GD1', 'GD2']
for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))
    p_best_run_df = best_runs_df[best_runs_df.player == player_name]
    input_vector = p_best_run_df[PS_REDUCED_COL_NAMES].iloc[0].values
    opt_info = OptInfo(input_vector)
    p_fnf_min_df = pd.read_csv(
        os.path.join(INTERIM_DATA_DIR, player_name, f'{player_name}_data_matrix_fav_non_fav_gd.csv')
    )

    all_state_dicts = opt_info.get_all_state_dict_tuple_list(is_longname=True)
    base_per_gd_dict = {}
    for gd in GD_STATES:
        gd_df = p_fnf_min_df[p_fnf_min_df.state == gd][['favorite', 'non-favorite']]
        gd_duration = gd_df.values.sum()
        energy_total = 0
        for state_name, state_dict in all_state_dicts:
            state_minutes = gd_df[state_name].iloc[0]
            energy_total += state_dict[str(gd)] * state_minutes
        reference_energy = energy_total / gd_duration
        base_per_gd_dict[gd] = reference_energy

    load_plt_style()
    plt.rc('ytick', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('axes', labelsize=14)

    # Create a heatmap of the transition matrix
    fig, ax = plt.subplots(figsize=(5, 2.3))
    ax.grid(False)
    colors = [(0.0, 'white'), (0.33, 'blue'), (0.66, 'gray'), (1.0, 'red')]
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

    fig_path = os.path.join(FIGURES_DIR, 'results')
    create_dir(fig_path)
    transition_matrix = np.zeros([len(GD_STATES), len(all_state_dicts)])
    for i, state_tuple in enumerate(all_state_dicts):
        state_name, state_vector = state_tuple
        s_vect = opt_info.get_state_vector(state_name=state_name)

        final_dict = {}
        for base, result in zip(base_per_gd_dict.items(), s_vect.items()):
            gd, base_value = base
            _, result_value = result
            final_dict[gd] = result_value / base_value
        transition_matrix[:, i] = list(final_dict.values())
    transition_matrix = np.array(transition_matrix)
    transition_matrix = np.transpose(transition_matrix)

    ax.imshow(transition_matrix, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1.5)
    # Add colorbar
    PCM = [PCM for PCM in ax.get_children() if isinstance(PCM, ScalarMappable)][0]
    if i > 1:
        cbar = plt.colorbar(PCM, ax=ax, fraction=0.047, pad=0.005)
        cbar.set_label('Transition Probability', rotation=270, labelpad=20)

    # Add labels to axes
    ax.set_xticks(np.arange(len(GD_STATES)), [f'GD$-${abs(l)}' if l < 0 else f'GD{l}' for l in GD_STATES])
    ax.set_yticks([0, 1], ['f', 'nf'])
    add_labels_to_cm(ax, transition_matrix)

    fig.savefig(os.path.join(fig_path, f'{player_name}_matrix_{save_name}.png'), dpi=300)
    plt.close()
