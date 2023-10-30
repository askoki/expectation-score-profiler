import os
import pandas as pd

from settings import INTERIM_DATA_DIR
from src.features.data_loaders import load_df_data

from src.features.file_helpers import create_dir
from src.visualization.constants import PGE_STATES, GD_STATES, GD_REDUCED_STATES, TRANSITION_STATES, REDUCED_PGE_STATES
from src.visualization.helpers.processing import init_state_df, change_state_df

df = load_df_data(is_dummy=False)

players_list = df.sort_values('athlete_id').athlete.unique()
players_count = len(players_list)

# favorite, close, underdog 9 states: TRANSITION_STATES
fcu_9state_df = init_state_df(is_fcu=True, is_gdc=True)
# favorite, non-favorite 9 states: TRANSITION_STATES
fnf_9state_df = init_state_df(is_fcu=False, is_gdc=True)
# favorite, close, underdog 4 states: GD_STATES
fcu_4state_df = init_state_df(is_fcu=True, is_gdc=False)
# favorite, non-favorite 4 states: GD_STATES
fnf_4state_df = init_state_df(is_fcu=False, is_gdc=False)

for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    save_path = os.path.join(INTERIM_DATA_DIR, player_name)
    create_dir(save_path)

    p_df = df[df.athlete == player_name]
    player_data_stats_df = pd.DataFrame()
    for last_gd, curr_gd in TRANSITION_STATES:
        minute_state_df = pd.DataFrame({
            'state': f'({last_gd}, {curr_gd})',
        }, index=[0])
        for expectation in PGE_STATES:
            pge_state_minutes = p_df[
                (p_df.betting_position == expectation) &
                (p_df.last_gd == last_gd) &
                (p_df.gd == curr_gd)
                ]
            minute_state_df.loc[:, expectation] = pge_state_minutes.duration_min.sum()
        player_data_stats_df = pd.concat([player_data_stats_df, minute_state_df])
    player_data_stats_df.reset_index(inplace=True, drop=True)
    player_data_stats_df.to_csv(os.path.join(save_path, f'{player_name}_data_matrix.csv'), index=False)
    fcu_9state_df = change_state_df(fcu_9state_df, player_data_stats_df)

    # GD States
    player_gd_stats_df = pd.DataFrame()
    for gd in GD_STATES:
        minute_state_df = pd.DataFrame({
            'state': gd,
        }, index=[0])
        for expectation in PGE_STATES:
            pge_state_minutes = p_df[
                (p_df.betting_position == expectation) &
                (p_df.gd == gd)
                ]
            minute_state_df.loc[:, expectation] = pge_state_minutes.duration_min.sum()
        player_gd_stats_df = pd.concat([player_gd_stats_df, minute_state_df])
    player_gd_stats_df.reset_index(inplace=True, drop=True)
    player_gd_stats_df.to_csv(os.path.join(save_path, f'{player_name}_data_matrix_gd.csv'), index=False)
    fcu_4state_df = change_state_df(fcu_4state_df, player_gd_stats_df)

    # GD reduced states
    player_gd_reduced_df = pd.DataFrame()
    for gd in GD_REDUCED_STATES:
        minute_state_df = pd.DataFrame({
            'state': gd,
        }, index=[0])
        for expectation in PGE_STATES:
            pge_state_minutes = p_df[
                (p_df.betting_position == expectation) &
                (p_df.gd == gd)
                ]
            minute_state_df.loc[:, expectation] = pge_state_minutes.duration_min.sum()
        player_gd_reduced_df = pd.concat([player_gd_reduced_df, minute_state_df])
    player_gd_reduced_df.reset_index(inplace=True, drop=True)
    player_gd_reduced_df.to_csv(os.path.join(save_path, f'{player_name}_data_matrix_gd_reduced.csv'), index=False)

    # ---------------------------------------------------------- Favorite-not favorite ----------------------------------------------------------
    # GD transition states
    player_expectation_reduced_df = pd.DataFrame()
    for last_gd, curr_gd in TRANSITION_STATES:
        minute_state_df = pd.DataFrame({
            'state': f'({last_gd}, {curr_gd})',
        }, index=[0])
        for expectation in REDUCED_PGE_STATES:
            expectation_search = ['close', 'underdog'] if expectation == 'non-favorite' else [expectation]
            pge_state_minutes = p_df[
                (p_df.betting_position.isin(expectation_search)) &
                (p_df.last_gd == last_gd) &
                (p_df.gd == curr_gd)
                ]
            minute_state_df.loc[:, expectation] = pge_state_minutes.duration_min.sum()
        player_expectation_reduced_df = pd.concat([player_expectation_reduced_df, minute_state_df])
    player_expectation_reduced_df.reset_index(inplace=True, drop=True)
    player_expectation_reduced_df.to_csv(
        os.path.join(save_path, f'{player_name}_data_matrix_expectation_fav_non_fav.csv'), index=False
    )
    fnf_9state_df = change_state_df(fnf_9state_df, player_expectation_reduced_df)

    # PGE reduced and states reduced
    player_red_pge_gd_df = pd.DataFrame()
    for gd in GD_STATES:
        minute_state_df = pd.DataFrame({
            'state': gd,
        }, index=[0])
        for expectation in REDUCED_PGE_STATES:
            expectation_search = ['close', 'underdog'] if expectation == 'non-favorite' else [expectation]
            pge_state_minutes = p_df[
                (p_df.betting_position.isin(expectation_search)) &
                (p_df.gd == gd)
                ]
            minute_state_df.loc[:, expectation] = pge_state_minutes.duration_min.sum()
        player_red_pge_gd_df = pd.concat([player_red_pge_gd_df, minute_state_df])
    player_red_pge_gd_df.reset_index(inplace=True, drop=True)
    player_red_pge_gd_df.to_csv(os.path.join(save_path, f'{player_name}_data_matrix_fav_non_fav_gd.csv'), index=False)
    fnf_4state_df = change_state_df(fnf_4state_df, player_red_pge_gd_df)

    # print(player_name)
    # print(player_red_pge_gd_df)
fcu_9state_df.to_csv(os.path.join(INTERIM_DATA_DIR, 'fcu_9states_all_players.csv'), index=False)
fnf_9state_df.to_csv(os.path.join(INTERIM_DATA_DIR, 'fnf_9states_all_players.csv'), index=False)
fcu_4state_df.to_csv(os.path.join(INTERIM_DATA_DIR, 'fcu_4state_all_players.csv'), index=False)
fnf_4state_df.to_csv(os.path.join(INTERIM_DATA_DIR, 'fnf_4state_all_players.csv'), index=False)
