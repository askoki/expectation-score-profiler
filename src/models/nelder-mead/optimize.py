import os
import time
import functools
import pandas as pd
from indago import NelderMead
from settings import USE_DUMMY, NELDER_MEAD_REPORTS_DIR
from src.features.file_helpers import create_dir
from src.features.optimisation.data_helpers import generate_initial_vector_reduced_approach
from src.features.optimisation.processing import PlayerDataProcessor, OptInfo, fun_min_reduced
from src.features.data_loaders import load_optimisation_data
from src.features.optimisation.results_helpers import post_iteration_processing
from src.features.optimisation.results_helpers import create_run_results_df
from src.features.utils import log, get_duration_hour_min_sec

from src.models.constants import NUM_RUNS, bounds_stamina, bounds_power

if __name__ == '__main__':
    data_holder = load_optimisation_data(is_dummy=USE_DUMMY)
    create_dir(NELDER_MEAD_REPORTS_DIR)
    players_list = data_holder.get_players()
    players_count = len(players_list)

    if __name__ == '__main__':
        for i, player_name in enumerate(players_list):
            log(f'Processing player: {player_name} {i + 1}/{players_count}')

            player_save_name = player_name.replace(' ', '_')
            save_path = os.path.join(NELDER_MEAD_REPORTS_DIR, player_save_name)
            create_dir(save_path)

            # OPTIMISATION PROCESS
            player_data_holder = PlayerDataProcessor(player_name, data_holder.get_player_data(player_name))
            results = []

            start = time.time()
            all_runs_df = pd.DataFrame()
            for j in range(NUM_RUNS):
                log(f'Run {j + 1} for all games')
                # init score
                init_vector = generate_initial_vector_reduced_approach()
                opt_desc = OptInfo(init_vector)
                input_vector = opt_desc.get_input_vector()

                optimizer = NelderMead()
                optimizer.variant = 'GaoHan'
                iteration_dict = {
                    'iteration': j,
                    'X': [],
                    'f': []
                }
                post_iteration_function = functools.partial(post_iteration_processing, iteration_dict=iteration_dict)
                optimizer.post_iteration_processing = post_iteration_function
                optimizer.number_of_processes = 'max'
                optimizer.objectives = 2
                optimizer.objective_labels = ['MSE', 'MAE']
                optimizer.objective_weights = [1.0, 0.0]
                optimizer.monitoring = 'none'
                optimizer.max_elapsed_time = 600
                optimizer.max_stalled_iterations = 100

                lb_vector = [bounds_power[0] for i in range(len(init_vector) - 1)]
                lb_vector.append(bounds_stamina[0])

                optimizer.dimensions = len(lb_vector)
                optimizer.lb = lb_vector
                ub_vector = [bounds_power[1] for i in range(len(init_vector) - 1)]
                ub_vector.append(bounds_stamina[1])
                optimizer.ub = ub_vector

                evaluation_function = functools.partial(
                    fun_min_reduced,
                    player_data_holder=player_data_holder
                )
                optimizer.evaluation_function = evaluation_function
                result = optimizer.optimize()
                results.append(result.X)

                opt_results = OptInfo(result.X)
                run_df = create_run_results_df(
                    it_count=j, init_params=str(input_vector), num_steps=optimizer.it, num_eval=optimizer.eval,
                    opt_results=opt_results, fin_cost_fun=result.f, fin_abs_err=result.O[1]
                )
                # inner run results
                reshape_size = optimizer.history['f'].shape[0]
                opt_variable_vectors_df = pd.DataFrame(optimizer.history['X'], columns=opt_results.get_input_names())
                opt_variable_vectors_df.to_csv(
                    os.path.join(save_path, f'run_{j}_opt_variable_steps.csv'), index=False
                )
                cost_function_df = pd.DataFrame(
                    optimizer.history['f'].reshape(1, reshape_size)[0], columns=['cost_fun']
                )
                cost_function_df.loc[:, 'num_eval'] = optimizer.history['eval'].reshape(1, reshape_size)[0]
                cost_function_df.to_csv(os.path.join(save_path, f'run_{j}_cost_fun_evaluations.csv'), index=False)
                all_runs_df = pd.concat([all_runs_df, run_df])
            all_runs_df.to_csv(os.path.join(save_path, 'all_runs_df.csv'), index=False)
            end = time.time()
            hours, minutes, seconds = get_duration_hour_min_sec(start=start, end=end)
            log(f'Time for optimisation: {hours}h {minutes}min {seconds}s')
