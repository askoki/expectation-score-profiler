import functools
import numpy as np

from typing import Tuple, List
from src.features.typing import PlayerMeasurement, GDEnergyDict, TransitionStateDict, GDPowerDict


class OptInfo:
    favorite_states: GDPowerDict = {}
    non_favorite_states: GDPowerDict = {}

    def __init__(self, input_list: list):
        self.favorite_states = {}
        self.non_favorite_states = {}
        all_keys = GDPowerDict.__dict__['__annotations__'].keys()
        vector_length = len(all_keys)
        for i, key_name in enumerate(all_keys):
            self.favorite_states[key_name] = input_list[i]
            self.non_favorite_states[key_name] = input_list[vector_length + i]
        self.stamina = input_list[-1]

    def get_all_state_dict_tuple_list(self, is_longname=False) -> List[Tuple[str, GDPowerDict]]:
        if is_longname:
            return [
                ('favorite', self.favorite_states),
                ('non-favorite', self.non_favorite_states),
            ]
        return [
            ('fav', self.favorite_states),
            ('non', self.non_favorite_states),
        ]

    def get_input_names(self, stamina_name: str = 'stamina') -> np.array:
        dict_names = []
        for state_name, state_dict in self.get_all_state_dict_tuple_list():
            dict_names.extend([f'{state_name}_{d}' for d in state_dict.keys()])
        dict_names.append(stamina_name)
        return np.array(dict_names)

    def get_state_vector(self, state_name: str) -> GDPowerDict:
        if state_name in ['favorite', 'fav']:
            return self.favorite_states
        elif state_name in ['non-favorite', 'non']:
            return self.non_favorite_states
        else:
            raise Exception(f'State={state_name} could not be found')

    def get_input_vector(self) -> np.array:
        feature_list = list(self.favorite_states.values())
        feature_list.extend(self.non_favorite_states.values())
        feature_list.append(self.stamina)
        return np.array(feature_list)


class PowerScore:
    favorite_states: TransitionStateDict = {}
    close_states: TransitionStateDict = {}
    underdog_states: TransitionStateDict = {}

    def __init__(self, input_list: list):
        self.favorite_states = {}
        self.close_states = {}
        self.underdog_states = {}
        all_keys = TransitionStateDict.__dict__['__annotations__'].keys()
        vector_length = len(all_keys)
        for i, key_name in enumerate(all_keys):
            self.favorite_states[key_name] = input_list[i]
            self.close_states[key_name] = input_list[vector_length + i]
            self.underdog_states[key_name] = input_list[(vector_length * 2) + i]
        self.stamina = input_list[-1]

    def get_all_state_dict_tuple_list(self, is_longname=False) -> List[Tuple[str, TransitionStateDict]]:
        if is_longname:
            return [
                ('favorite', self.favorite_states),
                ('close', self.close_states),
                ('underdog', self.underdog_states)
            ]
        return [
            ('fav', self.favorite_states),
            ('cls', self.close_states),
            ('und', self.underdog_states)
        ]

    def get_input_names(self, stamina_name: str = 'stamina') -> np.array:
        dict_names = []
        for state_name, state_dict in self.get_all_state_dict_tuple_list():
            dict_names.extend([f'{state_name}_{d}' for d in state_dict.keys()])
        dict_names.append(stamina_name)
        return np.array(dict_names)

    def get_state_vector(self, state_name: str) -> TransitionStateDict:
        if state_name in ['favorite', 'fav']:
            return self.favorite_states
        elif state_name in ['close', 'cls']:
            return self.close_states
        elif state_name in ['underdog', 'und']:
            return self.underdog_states
        else:
            raise Exception(f'State={state_name} could not be found')

    def get_input_vector(self) -> np.array:
        feature_list = list(self.favorite_states.values())
        feature_list.extend(self.close_states.values())
        feature_list.extend(self.underdog_states.values())
        feature_list.append(self.stamina)
        return np.array(feature_list)


def exponential_decay_fatigue_function(time: int, coefficient: float) -> float:
    return np.exp(-time * coefficient)


def calc_fatigue_effect(stamina_factor: float, min_from_start_array: np.array) -> np.array:
    FULL_GAME_DURATION = 90
    alpha = -np.log(stamina_factor) / FULL_GAME_DURATION
    calc_fatigue = functools.partial(exponential_decay_fatigue_function, coefficient=alpha)
    fatigue_influence = np.array(list(map(calc_fatigue, min_from_start_array)))
    return fatigue_influence


def calc_reduced_model_values(gd: np.array, opt_info: OptInfo, duration_min: np.array, min_from_start: np.array,
                              state: str):
    # first load base energy (median values per GD)
    calc_power = gd.copy()
    opt_dict = opt_info.get_state_vector(state_name=state)
    for gd_key in GDPowerDict.__dict__['__annotations__'].keys():
        power_gd = opt_dict[gd_key]
        calc_power[calc_power == int(gd_key)] = power_gd

    calc_energy = np.multiply(calc_power, duration_min)
    # finally add fatigue effect
    fatigue_effect = calc_fatigue_effect(opt_info.stamina, min_from_start)
    calc_energy = np.multiply(calc_energy, fatigue_effect)
    return calc_energy


def calc_model_values(gd: np.array,
                      transition_gd: np.array,
                      power_score: PowerScore,
                      base_energy: GDEnergyDict,
                      min_from_start: np.array,
                      state: str):
    # first load base energy (median values per GD)
    calc_energy = gd.copy()
    for str_score in base_energy.keys():
        base_value = base_energy[str_score]
        # case when player has no base energy for the given scoreline
        if np.isnan(base_value):
            continue
        int_score = int(str_score)
        calc_energy[calc_energy == int_score] = base_value
    # Now include coefficients
    energy_correction_coefs = transition_gd.copy()
    state_dict: TransitionStateDict = power_score.get_state_vector(state_name=state)
    for transition_state in state_dict.keys():
        energy_correction_coefs[energy_correction_coefs == transition_state] = state_dict[transition_state]
    # for starting case leave it with no effect
    energy_correction_coefs[energy_correction_coefs == '0_0'] = 1
    # outlier
    energy_correction_coefs[energy_correction_coefs == '0_2'] = state_dict['1_2']
    energy_correction_coefs[energy_correction_coefs == '-2_0'] = state_dict['-1_0']

    calc_energy = np.multiply(calc_energy, energy_correction_coefs)
    # finally add fatigue effect
    fatigue_effect = calc_fatigue_effect(power_score.stamina, min_from_start)
    calc_energy = np.multiply(calc_energy, fatigue_effect)
    return calc_energy


class PlayerDataProcessor:

    def __init__(self, player_name: str, player_measurements: PlayerMeasurement):
        self.player_name = player_name
        self.player_measurements = player_measurements

    def calculate_energy_for_all_matches(self, power_score: PowerScore) -> None:
        player_matches = self.player_measurements['values']
        for p_match in player_matches:
            calc_energy = calc_model_values(
                gd=p_match['gd'],
                transition_gd=p_match['transition_gd'],
                power_score=power_score,
                base_energy=self.player_measurements['base_energy'],
                min_from_start=p_match['min_from_start'],
                state=p_match['betting_position'][0]
            )
            p_match['calculated_values'] = calc_energy

    def calculate_energy_for_all_matches_reduced_approach(self, opt_info: OptInfo) -> None:
        player_matches = self.player_measurements['values']
        for p_match in player_matches:
            calc_energy = calc_reduced_model_values(
                gd=p_match['gd'],
                opt_info=opt_info,
                duration_min=p_match['duration_min'],
                min_from_start=p_match['min_from_start'],
                state=p_match['betting_position_reduced'][0]
            )
            p_match['calculated_values'] = calc_energy

    def get_error(self) -> float:
        player_matches = self.player_measurements['values']
        error = 0
        real_vals = []
        pred_vals = []
        for p_match in player_matches:
            real_vals = np.concatenate((real_vals, p_match['real_values']))
            pred_vals = np.concatenate((pred_vals, p_match['calculated_values']))
            match_error = (p_match['calculated_values'] - p_match['real_values']) ** 2
            error += match_error.sum()
        return error / self.player_measurements['total_minutes']

    def get_abs_error(self) -> float:
        player_matches = self.player_measurements['values']
        error = 0
        real_vals = []
        pred_vals = []
        for p_match in player_matches:
            real_vals = np.concatenate((real_vals, p_match['real_values']))
            pred_vals = np.concatenate((pred_vals, p_match['calculated_values']))
            match_error = np.abs(p_match['calculated_values'] - p_match['real_values'])
            error += match_error.sum()
        return error


def fun_min(input_vector: np.array, player_data_holder: PlayerDataProcessor) -> float:
    ps_descriptor = PowerScore(input_vector)
    player_data_holder.calculate_energy_for_all_matches(ps_descriptor)
    final_error = player_data_holder.get_error()
    return final_error


def fun_min_reduced(input_vector: np.array, player_data_holder: PlayerDataProcessor) -> Tuple[float, float]:
    ps_descriptor = OptInfo(input_vector)
    player_data_holder.calculate_energy_for_all_matches_reduced_approach(ps_descriptor)
    final_error = player_data_holder.get_error()
    abs_error = player_data_holder.get_abs_error()
    return final_error, abs_error
