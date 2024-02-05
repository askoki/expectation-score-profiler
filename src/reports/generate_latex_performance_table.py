import os
import sys
import pandas as pd
from typing import Tuple
from settings import REPORTS_DIR
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.data_loaders import load_optimisation_data
from src.visualization.helpers.processing import get_best_run
from src.features.optimisation.processing import OptInfo
from src.features.optimisation.processing import PlayerDataProcessor

data_holder = load_optimisation_data(is_dummy=False)
players_list = data_holder.get_players()
players_count = len(players_list)
eval_err_df = pd.DataFrame()

param_names = [
    'fav_-2', 'fav_-1', 'fav_0',
    'fav_1', 'fav_2', 'non_-2', 'non_-1', 'non_0', 'non_1', 'non_2',
    'stamina'
]


def calc_metrics(player_data_holder: PlayerDataProcessor, opt_info: OptInfo) -> Tuple[float, float, float]:
    predicted_values = []
    real_values = []
    player_data_holder.calculate_energy_for_all_matches_reduced_approach(opt_info)
    for match_data in player_data_holder.player_measurements['values']:
        predicted_values.extend(match_data['calculated_values'])
        real_values.extend(match_data['real_values'])
    mae = mean_absolute_error(real_values, predicted_values)
    mse = mean_squared_error(real_values, predicted_values)
    rmse = mean_squared_error(real_values, predicted_values, squared=False)
    return mae, mse, rmse


for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    methods = ['pso', 'nelder-mead']
    for method in methods:
        save_path = os.path.join(REPORTS_DIR, method, player_name)
        all_df = pd.read_csv(os.path.join(save_path, 'all_runs_df.csv'))
        best_it_df = get_best_run(all_df)

        player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))
        final_vector = best_it_df[param_names].values
        opt_info = OptInfo(final_vector)

        mae, mse, rmse = calc_metrics(player_data_holder, opt_info)

        b1_mean_vector = [
            *list(player_data_holder.player_measurements['base_energy_mean']['favorite'].values()),
            *list(player_data_holder.player_measurements['base_energy_mean']['non-favorite'].values()),
            1
        ]
        b1_opt_info = OptInfo(b1_mean_vector)
        b1_mae, b1_mse, b1_rmse = calc_metrics(player_data_holder, b1_opt_info)

        b2_median_vector = [
            *list(player_data_holder.player_measurements['base_energy_median']['favorite'].values()),
            *list(player_data_holder.player_measurements['base_energy_median']['non-favorite'].values()),
            1
        ]
        b2_opt_info = OptInfo(b2_median_vector)
        b2_mae, b2_mse, b2_rmse = calc_metrics(player_data_holder, b2_opt_info)

        player_eval_err = pd.DataFrame({
            'player': player_name,
            'method': method,
            'mae': mae,
            'b1_mae': b1_mae,
            'b2_mae': b2_mae,
            'mse': mse,
            'b1_mse': b1_mse,
            'b2_mse': b2_mse,
            'rmse': rmse,
            'b1_rmse': b1_rmse,
            'b2_rmse': b2_rmse,
            'num_minutes': player_data_holder.player_measurements['total_minutes']
        }, index=[0])
        eval_err_df = pd.concat([eval_err_df, player_eval_err])
    eval_err_df.reset_index(inplace=True, drop=True)
eval_err_df.reset_index(inplace=True, drop=True)

eval_err_df.loc[:, 'athlete_id'] = eval_err_df.player.apply(lambda r: r.split('athlete')[1])
eval_err_df.loc[:, 'athlete_id'] = eval_err_df.loc[:, 'athlete_id'].astype(int)
eval_err_df = eval_err_df.sort_values('athlete_id')

nel_df = eval_err_df[eval_err_df.method == 'nelder-mead'].round(5)
mean_df = nel_df.mean(numeric_only=True)
std_df = nel_df.std(numeric_only=True)

# -------------------- PERFORMANCE TABLE -------------------------
performance_list = [
    ('', 'Model'),
    ('b1_', '$B_1$'),
    ('b2_', '$B_2$'),
]
with open(os.path.join(REPORTS_DIR, 'performance_overall_table.tex'), 'w') as f:
    sys.stdout = f
    for key_prefix, label in performance_list:
        print('\midrule')
        # metrics = ['mae', 'rmse', 'mse']
        # for metric_name in metrics:
        print(
            f'{label} & ${round(mean_df[f"{key_prefix}mae"], 2)}~~~~{round(std_df[f"{key_prefix}mae"], 2)}$ & ${round(mean_df[f"{key_prefix}rmse"], 2)}~~~~{round(std_df[f"{key_prefix}rmse"], 2)}$ & ${round(mean_df[f"{key_prefix}mse"], 2)}~~~~{round(std_df[f"{key_prefix}mse"], 2)}$ \\\\'
        )
    print('\midrule')
sys.stdout = sys.__stdout__
