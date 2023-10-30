import os
import pandas as pd
from argparse import ArgumentParser

from settings import REPORTS_DIR, PSO_REPORTS_DIR, NELDER_MEAD_REPORTS_DIR

parser = ArgumentParser(description='Please provide data source info')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--pso', dest='is_pso', action='store_true')
group.add_argument('--nm', dest='is_pso', action='store_false')

is_pso = parser.parse_args().is_pso
source_path = PSO_REPORTS_DIR if is_pso else NELDER_MEAD_REPORTS_DIR
save_name = 'pso' if is_pso else 'nm'

all_runs_df = pd.DataFrame()
for player_dir in os.listdir(source_path):
    player_all_runs_df = pd.read_csv(os.path.join(source_path, player_dir, 'all_runs_df.csv'))
    # Return it to an original format, so it is the same as load_data
    player_all_runs_df.loc[:, 'player'] = player_dir.replace('_', ' ')
    all_runs_df = pd.concat([all_runs_df, player_all_runs_df])
all_runs_df.reset_index(inplace=True, drop=True)
best_run_df = all_runs_df.sort_values('final_cost_function').groupby('player').agg('first')
best_run_df = best_run_df.reset_index()

all_runs_df.to_csv(os.path.join(REPORTS_DIR, f'{save_name}_all_runs_df.csv'), index=False)
best_run_df.to_csv(os.path.join(REPORTS_DIR, f'{save_name}_best_run_df.csv'), index=False)
