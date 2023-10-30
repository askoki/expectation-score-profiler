from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features.typing import TransitionStateDict, GDPowerDict
from src.visualization.constants import GD_STATES, TRANSITION_STATES


def parse_all_iterations_df(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, 'resulting_vector'] = df.loc[:, 'resulting_vector'].str.replace('\n', '', regex=True)
    df.loc[:, 'resulting_vector'] = df.loc[:, 'resulting_vector'].str.replace('\[', '', regex=True)
    df.loc[:, 'resulting_vector'] = df.loc[:, 'resulting_vector'].str.replace('\]', '', regex=True)
    return df


def get_best_run(all_it_df: pd.DataFrame) -> pd.Series:
    return all_it_df.sort_values(['final_cost_function', 'number_of_evaluations'], ascending=True).iloc[0]


GD_STATES_LIST = np.arange(-2, 3)


def get_init_state_df() -> pd.DataFrame:
    state_df = pd.DataFrame()
    for gd in GD_STATES_LIST:
        s_df = pd.DataFrame({
            'code': f'{gd}',
            'label': f'GD{gd}',
            'value': 0
        }, index=[0])
        state_df = pd.concat([state_df, s_df])
    state_df.reset_index(inplace=True, drop=True)
    return state_df


def create_transition_matrix(state_dict: TransitionStateDict) -> np.array:
    state_df = get_init_state_df()
    for key, val in state_dict.items():
        state_df.loc[state_df.code == key, 'value'] = val

    transition_matrix = []
    row_length = len(GD_STATES_LIST)
    for i in range(0, state_df.shape[0], row_length):
        transition_matrix.append(state_df.value.iloc[i:i + row_length])
    return np.array(transition_matrix)


def create_state_df(state_dict: GDPowerDict) -> pd.DataFrame:
    state_df = get_init_state_df()
    for key, val in state_dict.items():
        state_df.loc[state_df.code == key, 'value'] = val

    return state_df


def add_labels_to_cm(ax: plt.Axes, transition_matrix: np.array) -> None:
    # Add labels, including the "-" for unspecified values
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if transition_matrix[i, j] != 0:
                ax.text(
                    j, i, str(transition_matrix[i, j].round(2)), ha='center', va='center', fontsize=12, color='white'
                )
            else:
                ax.text(j, i, "-", ha='center', va='center', fontsize=12, color='gray')


def init_state_df(is_fcu: bool, is_gdc: bool) -> pd.DataFrame:
    states = TRANSITION_STATES if is_gdc else GD_STATES
    df = pd.DataFrame()
    df.loc[:, 'state'] = states
    df.loc[:, 'favorite'] = 0
    if is_fcu:
        df.loc[:, 'close'] = 0
        df.loc[:, 'underdog'] = 0
    else:
        df.loc[:, 'non-favorite'] = 0
    return df


def change_state_df(state_df: pd.DataFrame, player_min_stats_df: pd.DataFrame,
                    minimum_min_threshold=10) -> pd.DataFrame:
    is_fcu = 'underdog' in player_min_stats_df.columns
    cols = ['favorite', 'close', 'underdog'] if is_fcu else ['favorite', 'non-favorite']
    for col in cols:
        state_df.loc[:, col] += player_min_stats_df.apply(
            lambda r: 1 if r[col] > minimum_min_threshold else 0,
            axis=1
        )
    return state_df


def convert_gd_to_step_fill_function(t_arr: np.array, gd_arr: np.array) -> Tuple[np.array, np.array]:
    t_ret_arr = []
    gd_ret_arr = []
    i = 1
    for t, gd in zip(t_arr[1:], gd_arr[1:]):
        if gd != gd_arr[i - 1]:
            t_ret_arr.append(t_arr[i - 1])
            gd_ret_arr.append(gd)
        t_ret_arr.append(t)
        gd_ret_arr.append(gd)
        i += 1
    return np.array(t_ret_arr), np.array(gd_ret_arr)