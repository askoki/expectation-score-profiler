import os
import pandas as pd
from argparse import ArgumentParser

from settings import REPORTS_DIR, PSO_REPORTS_DIR, NELDER_MEAD_REPORTS_DIR
from src.features.data_loaders import load_optimisation_data
from src.features.optimisation.processing import PlayerDataProcessor, OptInfo, fun_min_reduced

parser = ArgumentParser(description='Please provide data source info')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--pso', dest='is_pso', action='store_true')
group.add_argument('--nm', dest='is_pso', action='store_false')

is_pso = parser.parse_args().is_pso
source_path = PSO_REPORTS_DIR if is_pso else NELDER_MEAD_REPORTS_DIR
save_name = 'pso' if is_pso else 'nm'

best_run_df = pd.read_csv(os.path.join(REPORTS_DIR, f'{save_name}_best_run_df.csv'))
data_holder = load_optimisation_data(is_dummy=False)
players_list = data_holder.get_players()
players_count = len(players_list)

comp_df = pd.DataFrame()
for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))
    p_best_it_df = best_run_df[best_run_df.player == player_name]

    input_vector = p_best_it_df[
        [
            'fav_-2', 'fav_-1', 'fav_0', 'fav_1', 'fav_2', 'non_-2', 'non_-1',
            'non_0', 'non_1', 'non_2', 'stamina'
        ]
    ].iloc[0].values
    opt_info = OptInfo(input_vector)
    opt_error, opt_abs_err = fun_min_reduced(input_vector, player_data_holder)

    base_input_vector_mean = list(player_data_holder.player_measurements['base_energy_mean']['favorite'].values())
    base_input_vector_mean.extend(
        list(player_data_holder.player_measurements['base_energy_mean']['non-favorite'].values()))
    base_input_vector_mean.append(1)

    base_input_vector_median = list(player_data_holder.player_measurements['base_energy_median']['favorite'].values())
    base_input_vector_median.extend(
        list(player_data_holder.player_measurements['base_energy_median']['non-favorite'].values())
    )
    base_input_vector_median.append(1)
    total_energy_spent = 0
    for match_data in player_data_holder.player_measurements['values']:
        total_energy_spent += match_data['real_values'].sum()

    baseline_error_mean, b1_abs_err = fun_min_reduced(base_input_vector_mean, player_data_holder)
    baseline_error_median, b2_abs_err = fun_min_reduced(base_input_vector_median, player_data_holder)

    player_comp_df = pd.DataFrame({
        'method': save_name,
        'player': player_name,
        'opt_error': opt_error,
        'baseline_error_mean': baseline_error_mean,
        'baseline_error_median': baseline_error_median,
        'total_energy_spent': total_energy_spent,
        'opt_abs': opt_abs_err,
        'b1_abs': b1_abs_err,
        'b2_abs': b2_abs_err,
        'opt_%_err': opt_abs_err / total_energy_spent,
        'b1_%_err': b1_abs_err / total_energy_spent,
        'b2_%_err': b2_abs_err / total_energy_spent,
        'baseline_err_mean/opt_err': baseline_error_mean / opt_error,
        'baseline_error_median/opt_err': baseline_error_median / opt_error,
    }, index=[0])
    comp_df = pd.concat([comp_df, player_comp_df])
comp_df.reset_index(inplace=True, drop=True)
comp_df.to_csv(
    os.path.join(
        os.path.join(REPORTS_DIR, f'{save_name}_comp_with_baseline_df.csv')
    ),
    index=False
)
