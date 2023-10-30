import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from src.features.data_loaders import load_optimisation_data
from src.features.file_helpers import create_dir
from src.features.optimisation.processing import PlayerDataProcessor, OptInfo
from src.features.typing import MplRadarPlotColors
from src.models.constants import bounds_stamina, bounds_power
from src.visualization.constants import MPL_GREEN, MPL_DARK_GREEN, MPL_ORANGE, \
    MPL_DARK_ORANGE, MPL_DARK_BLUE, LIGHT_BLUE, PS_REDUCED_COL_NAMES
from src.visualization.helpers.matplotlib_style import load_plt_style
from src.visualization.helpers.processing import create_state_df
from src.visualization.plots.radar_helpers import create_mpl_radar_color
from src.visualization.plots.radar_plots import plot_mpl_radar
from settings import REPORTS_DIR, ETA_SIGN, FIGURES_DIR, USE_DUMMY

# keep clean terminal
warnings.simplefilter(action='ignore', category=FutureWarning)

data_holder = load_optimisation_data(is_dummy=USE_DUMMY)
best_pso_run_df = pd.read_csv(os.path.join(REPORTS_DIR, f'pso_best_run_df.csv'))
best_nm_run_df = pd.read_csv(os.path.join(REPORTS_DIR, f'nm_best_run_df.csv'))
players_list = data_holder.get_players()
players_count = len(players_list)


def parse_graph_labels_and_values(df: pd.DataFrame) -> Tuple[list, list]:
    df.loc[:, 'label'] = df.apply(
        lambda
            r: f'${r.label}$' if not r.team_state else f'$P_{{{r.code},{r.team_state.capitalize()[0]}}}$',
        axis=1
    )
    df = df.dropna(subset=['value'])
    df.reset_index(inplace=True, drop=True)

    values = df.value.values.tolist()
    labels = df.label.values
    return values, labels


def get_stamina_row_df(ps: OptInfo) -> pd.DataFrame:
    return pd.DataFrame({
        'code': 'stamina',
        'label': ETA_SIGN,
        'value': ps.stamina,
        'team_state': None
    }, index=[0])


comp_df = pd.DataFrame()
load_plt_style(is_default=True)
state_labels = ['GD-2', 'GD-1', 'GD0', 'GD1', 'GD2']
for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))
    p_best_pso_run_df = best_pso_run_df[best_pso_run_df.player == player_name]
    p_best_nm_run_df = best_nm_run_df[best_nm_run_df.player == player_name]
    input_vector_pso = p_best_pso_run_df[PS_REDUCED_COL_NAMES].iloc[0].values
    input_vector_nm = p_best_nm_run_df[PS_REDUCED_COL_NAMES].iloc[0].values

    power_score_pso = OptInfo(input_vector_pso)
    power_score_nm = OptInfo(input_vector_nm)

    all_pso_state_dicts = power_score_pso.get_all_state_dict_tuple_list(is_longname=False)
    all_nm_state_dicts = power_score_nm.get_all_state_dict_tuple_list(is_longname=False)
    all_state_dicts = zip(all_pso_state_dicts, all_nm_state_dicts)


    pso_df = pd.DataFrame()
    nm_df = pd.DataFrame()
    for i, method_tuple in enumerate(all_state_dicts):
        pso_state_name, pso_state_vector = method_tuple[0]
        nm_state_name, nm_state_vector = method_tuple[1]

        s_vect_pso = power_score_pso.get_state_vector(state_name=pso_state_name)
        result_pso_df = create_state_df(s_vect_pso)
        result_pso_df.loc[result_pso_df.value == 0, 'value'] = None
        result_pso_df = result_pso_df.dropna()

        s_vect_nm = power_score_nm.get_state_vector(state_name=nm_state_name)
        result_nm_df = create_state_df(s_vect_nm)
        result_nm_df.loc[result_nm_df.value == 0, 'value'] = None
        result_nm_df = result_nm_df.dropna()

        result_pso_df.loc[:, 'team_state'] = pso_state_name
        pso_df = pd.concat([pso_df, result_pso_df])
        result_nm_df.loc[:, 'team_state'] = nm_state_name
        nm_df = pd.concat([nm_df, result_nm_df])
    pso_stamina_row = get_stamina_row_df(power_score_pso)
    pso_df = pd.concat([pso_df, pso_stamina_row])

    nm_stamina_row = get_stamina_row_df(power_score_nm)
    nm_df = pd.concat([nm_df, nm_stamina_row])

    values, labels = parse_graph_labels_and_values(pso_df)
    values_nm, labels_nm = parse_graph_labels_and_values(nm_df)

    bound_low = [bounds_power[0]] * len(values)
    bound_high = [bounds_power[1]] * len(values)
    bound_low[-1] = bounds_stamina[0]
    bound_high[-1] = bounds_stamina[1]

    colors_pso: MplRadarPlotColors = {
        'rings_inner': create_mpl_radar_color(facecolor=LIGHT_BLUE, edgecolor=MPL_DARK_BLUE),
        'radar': create_mpl_radar_color(facecolor=MPL_GREEN, edgecolor=MPL_DARK_GREEN),
    }
    round_int = [True] * len(labels)
    round_int[-1] = False
    fig, ax = plot_mpl_radar(
        values, labels, bound_low=bound_low, bound_high=bound_high,
        colors=colors_pso, round_int=[round_int]
    )
    save_path = os.path.join(FIGURES_DIR, 'results')
    create_dir(save_path)
    fig.savefig(os.path.join(save_path, f'{player_name}_pso_results_radar.png'), dpi=300)
    plt.close()

    colors_nm: MplRadarPlotColors = {
        'rings_inner': create_mpl_radar_color(facecolor=LIGHT_BLUE, edgecolor=MPL_DARK_BLUE),
        'radar': create_mpl_radar_color(facecolor=MPL_ORANGE, edgecolor=MPL_DARK_ORANGE),
    }
    fig, ax = plot_mpl_radar(
        values_nm, labels_nm, bound_low=bound_low, bound_high=bound_high,
        colors=colors_nm, round_int=[round_int]
    )
    fig.savefig(os.path.join(save_path, f'{player_name}_nm_results_radar.png'), dpi=300)
    plt.close()
