import math
import os
import sys

import pandas as pd
from settings import REPORTS_DIR

from src.features.data_loaders import load_optimisation_data
from src.visualization.helpers.processing import get_best_run
from src.features.optimisation.processing import fun_min_reduced
from src.features.optimisation.processing import PlayerDataProcessor

data_holder = load_optimisation_data(is_dummy=False)
players_list = data_holder.get_players()
players_count = len(players_list)
eval_err_df = pd.DataFrame()
eval_run_df = pd.DataFrame()

param_names = [
    'fav_-2', 'fav_-1', 'fav_0',
    'fav_1', 'fav_2', 'non_-2', 'non_-1', 'non_0', 'non_1', 'non_2',
    'stamina'
]

for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    methods = ['pso', 'nelder-mead']
    for method in methods:
        save_path = os.path.join(REPORTS_DIR, method, player_name)
        all_df = pd.read_csv(os.path.join(save_path, 'all_runs_df.csv'))
        best_it_df = get_best_run(all_df)

        player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))
        final_vector = best_it_df[param_names].values
        model_error, model_abs_error = fun_min_reduced(final_vector, player_data_holder)

        b1_mean_vector = [
            *list(player_data_holder.player_measurements['base_energy_mean']['favorite'].values()),
            *list(player_data_holder.player_measurements['base_energy_mean']['non-favorite'].values()),
            1
        ]
        b1_error, b1_abs_error = fun_min_reduced(b1_mean_vector, player_data_holder)

        b2_median_vector = [
            *list(player_data_holder.player_measurements['base_energy_median']['favorite'].values()),
            *list(player_data_holder.player_measurements['base_energy_median']['non-favorite'].values()),
            1
        ]
        b2_error, b2_abs_error = fun_min_reduced(b2_median_vector, player_data_holder)
        player_eval_err = pd.DataFrame({
            'player': player_name,
            'method': method,
            'n_eval': best_it_df.number_of_evaluations,
            'cost_fun': best_it_df.final_cost_function,
            'b1_err': b1_error,
            'b2_err': b2_error,
            'model_abs_err': model_abs_error,
            'b1_abs_err': b1_abs_error,
            'b2_abs_err': b2_abs_error,
            'abs_vs_b1%': (b1_abs_error - model_abs_error) / b1_abs_error * 100,
            'abs_vs_b2%': (b2_abs_error - model_abs_error) / b2_abs_error * 100,
        }, index=[0])
        eval_err_df = pd.concat([eval_err_df, player_eval_err])
        # --------- run error ----------
        run = 0
        for idx, p_df in all_df.iterrows():
            final_run_vector = p_df[param_names].values
            model_run_err, model_run_abs_err = fun_min_reduced(final_run_vector, player_data_holder)
            run += 1
            player_eval_err = pd.DataFrame({
                'player': player_name,
                'run': run,
                'method': method,
                'n_eval': p_df.number_of_evaluations,
                'cost_fun': model_run_err,
                'b1_err': b1_error,
                'b2_err': b2_error,
                'model_abs_err': model_run_abs_err,
            }, index=[0])
            eval_run_df = pd.concat([eval_run_df, player_eval_err])
    eval_err_df.reset_index(inplace=True, drop=True)
eval_err_df.reset_index(inplace=True, drop=True)
eval_run_df.reset_index(inplace=True, drop=True)

eval_err_df.loc[:, 'athlete_id'] = eval_err_df.player.apply(lambda r: r.split('athlete')[1])
eval_err_df.loc[:, 'athlete_id'] = eval_err_df.loc[:, 'athlete_id'].astype(int)
eval_err_df = eval_err_df.sort_values('athlete_id')

with open(os.path.join(REPORTS_DIR, 'model_performance_per_player_table.tex'), 'w') as f:
    sys.stdout = f
    for idx, ath_group in eval_err_df.groupby('athlete_id'):
        ath_pso = ath_group[ath_group.method == 'pso'].round(5).iloc[0]
        ath_nel = ath_group[ath_group.method == 'nelder-mead'].round(5).iloc[0]
        print('\midrule')
        if ath_pso.cost_fun < ath_nel.cost_fun:
            print(
                f'{ath_pso.player} & {ath_pso.n_eval} & {ath_nel.n_eval} & \\textbf{{{round(ath_pso.cost_fun, 5)}}} & ${round(ath_nel.cost_fun, 5)}$ & ${round(ath_nel.b1_err, 2)}$ & ${round(ath_nel.b2_err, 2)}$ \\\\'
            )
        elif ath_pso.cost_fun > ath_nel.cost_fun:
            print(
                f'{ath_pso.player} & {ath_pso.n_eval} & {ath_nel.n_eval} & ${round(ath_pso.cost_fun, 5)}$ & \\textbf{{{round(ath_nel.cost_fun, 5)}}} & ${round(ath_nel.b1_err, 2)}$ & ${round(ath_nel.b2_err, 2)}$ \\\\'
            )
        else:
            print(
                f'{ath_pso.player} & {ath_pso.n_eval} & {ath_nel.n_eval} & \\textbf{{{round(ath_pso.cost_fun, 5)}}} & \\textbf{{{round(ath_nel.cost_fun, 5)}}} & ${round(ath_nel.b1_err, 2)}$ & ${round(ath_nel.b2_err, 2)}$ \\\\'
            )
sys.stdout = sys.__stdout__

# -------------------- ABSOLUTE ERROR -------------------------

with open(os.path.join(REPORTS_DIR, 'abs_error_per_player_table.tex'), 'w') as f:
    sys.stdout = f
    for idx, ath_group in eval_err_df.groupby('athlete_id'):
        ath_nel = ath_group[ath_group.method == 'nelder-mead'].round(5).iloc[0]
        print('\midrule')
        if ath_nel.model_abs_err < ath_nel.b1_abs_err:
            print(
                f'{ath_nel.player} & \\textbf{{{round(ath_nel.model_abs_err, 2)}}} & ${round(ath_nel.b1_abs_err, 2)}$ & ${round(ath_nel.b2_abs_err, 2)}$ & \\textbf{{-{round(ath_nel["abs_vs_b1%"], 2)}\%}} & \\textbf{{-{round(ath_nel["abs_vs_b2%"], 2)}\%}} \\\\'
            )
        elif ath_nel.b1_abs_err < ath_nel.b2_abs_err:
            print(
                f'{ath_nel.player} & ${round(ath_nel.model_abs_err, 2)}$ & \\textbf{{{round(ath_nel.b1_abs_err, 2)}}} & ${round(ath_nel.b2_abs_err, 2)}$ & +{round(ath_nel["abs_vs_b1%"], 2)}\% & +{round(ath_nel["abs_vs_b2%"], 2)}\% \\\\'
            )
        else:
            print(
                f'{ath_nel.player} & ${round(ath_nel.model_abs_err, 2)}$ & ${round(ath_nel.b1_abs_err, 2)}$ & \\textbf{{{round(ath_nel.b2_abs_err, 2)}}} & +{round(ath_nel["abs_vs_b1%"], 2)}\% & +{round(ath_nel["abs_vs_b2%"], 2)}\% \\\\'
            )
    ovr_df = eval_err_df[eval_err_df.method == 'nelder-mead']
    print('\midrule')
    print(
        f' & & & & $\mu$=-{round(ovr_df["abs_vs_b1%"].mean(), 2)}\% & $\mu$=-{round(ovr_df["abs_vs_b2%"].mean(), 2)}\% \\\\'
    )
sys.stdout = sys.__stdout__

# ---------------- VARIATION THROUGH RUNS --------------
eval_run_df.loc[:, 'athlete_id'] = eval_run_df.player.apply(lambda r: r.split('athlete')[1])
eval_run_df.loc[:, 'athlete_id'] = eval_run_df.loc[:, 'athlete_id'].astype(int)
eval_run_df = eval_run_df.sort_values('athlete_id')

eit_df = eval_run_df.copy()
mean_df = eit_df.groupby(['player', 'method']).agg({'cost_fun': 'mean', 'n_eval': 'mean'}).reset_index().rename(
    columns={'cost_fun': 'cost_err_mean', 'n_eval': 'n_eval_mean'}
)
std_df = eit_df.groupby(['player', 'method']).agg({'cost_fun': 'std', 'n_eval': 'std'}).reset_index().rename(
    columns={'cost_fun': 'cost_err_std', 'n_eval': 'n_eval_std'}
)
eit_df = mean_df.merge(std_df, on=['player', 'method'])
eit_df.loc[:, 'athlete_id'] = eit_df.player.apply(lambda r: r.split('athlete')[1])
eit_df.loc[:, 'athlete_id'] = eit_df.loc[:, 'athlete_id'].astype(int)
eit_df = eit_df.sort_values('athlete_id')


def format_number(number: float):
    if number == 0:
        return 0

    if number <= 1e6:
        return round(number, 2)
    exponent = int(math.log10(abs(number)))
    mantissa = number / (10 ** exponent)
    formatted_number = f"${mantissa:.2f} \cdot 10^{{{exponent}}}$"
    return formatted_number


with open(os.path.join(REPORTS_DIR, 'variation_through_runs_table.tex'), 'w') as f:
    sys.stdout = f

    for idx, ath_group in eit_df.groupby('athlete_id'):
        ath_pso = ath_group[ath_group.method == 'pso'].iloc[0]
        ath_pso.cost_err_mean = ath_pso.cost_err_mean.round(2)
        ath_pso.n_eval_mean = ath_pso.n_eval_mean.astype(int)
        ath_pso.n_eval_std = ath_pso.n_eval_std.astype(int)

        ath_nel = ath_group[ath_group.method == 'nelder-mead'].iloc[0]
        ath_nel.cost_err_mean = ath_nel.cost_err_mean.round(2)
        ath_nel.n_eval_mean = ath_nel.n_eval_mean.astype(int)
        ath_nel.n_eval_std = ath_nel.n_eval_std.astype(int)
        print('\midrule')
        print(
            f'{ath_pso.player} & {ath_pso.cost_err_mean} ${format_number(ath_pso.cost_err_std)}$ & {ath_nel.cost_err_mean} ${format_number(ath_nel.cost_err_std)}$ & {ath_pso.n_eval_mean} ${ath_pso.n_eval_std}$ & {ath_nel.n_eval_mean} ${ath_nel.n_eval_std}$\\\\'
        )
sys.stdout = sys.__stdout__
