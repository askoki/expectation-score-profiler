import time
import numpy as np
import pandas as pd
from typing import List
from random import randint, uniform
from src.features.utils import log
from src.features.typing import GDPowerDict, BaselineEnergyDict
from src.models.constants import bounds_stamina, bounds_power
from src.features.typing import PlayerMeasurement, PlayerMatchMeasurement
from src.visualization.constants import GD_STATES


class DataHolder:
    player_measurements: dict = {}

    def __init__(self, df: pd.DataFrame):
        data_load_start = time.time()
        all_players = df.sort_values('athlete_id').athlete.unique()
        for player in all_players:
            player_matches_df = df[df.athlete == player]
            player_matches_df.loc[:, 'betting_pos_red'] = player_matches_df.betting_position.str.replace(
                'underdog',
                'non-favorite'
            )
            player_matches_df.loc[:, 'betting_pos_red'] = player_matches_df.betting_pos_red.str.replace(
                'close',
                'non-favorite'
            )
            self.player_measurements[player]: PlayerMeasurement = {
                'values': [],
                'total_minutes': 0,
                'num_matches': player_matches_df.date.unique().shape[0],
                'base_energy_median': self._get_gd_base_energy(player_matches_df, 'median'),
                'base_energy_mean': self._get_gd_base_energy(player_matches_df, 'mean'),
            }
            for idx, match_df in player_matches_df.groupby('date'):
                num_minutes = match_df.duration_min.sum()
                match_df.loc[:, 'transition_gd'] = match_df.apply(lambda r: f'{r.last_gd}_{r.gd}', axis=1)
                match_dict: PlayerMatchMeasurement = {
                    'min_from_start': match_df.min_from_start.values,
                    'duration_min': match_df.duration_min.values,
                    'energy_per_min': match_df.energy_per_min.values,
                    'gd': match_df.gd.values,
                    'transition_gd': match_df.transition_gd.values,
                    'last_gd': match_df.last_gd.values,
                    'betting_position': match_df.betting_position.values,
                    'betting_position_reduced': match_df.betting_pos_red.values,
                    'real_values': match_df['energy (J/kg)'].values,
                    'calculated_values': [],
                    'num_minutes': num_minutes,
                    'game_label': match_df.date.iloc[0]
                }
                self.player_measurements[player]['values'].append(match_dict)
                self.player_measurements[player]['total_minutes'] += num_minutes
        data_load_end = time.time()
        log(f'Time for data load {data_load_end - data_load_start}')

    def _get_base_by_method_and_state(self, player_df: pd.DataFrame, method: str, state: str):
        state_filter_df = player_df[player_df.betting_pos_red == state]
        return_dict = {}
        for gd in GD_STATES:
            filter_df = state_filter_df[state_filter_df.gd == gd]
            if filter_df.shape[0] <= 0:
                return_dict[gd] = 0
                continue
            return_dict[gd] = getattr(filter_df['energy_per_min'], method)()
        return return_dict

    def _get_gd_base_energy(self, player_df: pd.DataFrame, method: str) -> BaselineEnergyDict:
        return {
            'favorite': self._get_base_by_method_and_state(player_df, method, 'favorite'),
            'non-favorite': self._get_base_by_method_and_state(player_df, method, 'non-favorite')
        }

    def get_players(self) -> list:
        return [*self.player_measurements.keys()]

    def get_player_data(self, player_name: str) -> PlayerMeasurement:
        return self.player_measurements[player_name]


def generate_initial_vector_reduced_approach() -> np.array:
    gd_vector_length = len(GDPowerDict.__dict__['__annotations__'].keys())
    # transition_vector_length * 2 for favorite and non-favorite
    init_vect: List[float] = [
        uniform(bounds_power[0], bounds_power[1]) for _ in range(gd_vector_length * 2)
    ]
    init_stamina_factor = randint(bounds_stamina[0] * 10, bounds_stamina[1] * 10) / 10.0
    init_vect.append(init_stamina_factor)
    return init_vect
