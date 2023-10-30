import pandas as pd

from src.features.optimisation.processing import PowerScore, OptInfo


def post_iteration_processing(it, candidates, best, iteration_dict: {}) -> None:
    if candidates[0] <= best:
        iteration_dict['X'].append(candidates[0].X)
        iteration_dict['f'].append(candidates[0].f)
    return


def create_run_results_df(it_count: int, init_params: str, num_steps: int, num_eval: int,
                          opt_results: PowerScore or OptInfo, fin_cost_fun: int, fin_abs_err: int) -> pd.DataFrame:
    res_df = pd.DataFrame({
        'iteration_count': it_count,
        'init_params': init_params,
        'num_steps': num_steps,
        'number_of_evaluations': num_eval,
        'final_cost_function': fin_cost_fun,
        'final_abs_error': fin_abs_err
    }, index=[it_count])
    for state_name, state_dict in opt_results.get_all_state_dict_tuple_list():
        for key, val in state_dict.items():
            res_df.loc[:, f'{state_name}_{key}'] = val
    res_df.loc[:, 'stamina'] = opt_results.stamina
    return res_df
