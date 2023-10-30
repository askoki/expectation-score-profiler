import numpy as np
import matplotlib.pyplot as plt
from src.visualization.helpers.matplotlib_style import load_plt_style


def plot_energy_comparison(
        min_from_start: np.array,
        energy_real: np.array,
        energy_calc: np.array,
        gd_arr: np.array,
        correction_coef_arr: np.array,
        pge_state: str,
) -> plt.Figure:
    load_plt_style()
    linewidth = 3

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
    # fig.patch.set_facecolor('white')

    # ax1.set_facecolor('xkcd:white')
    ax1.set_title(f'Energy comparison')
    ax1.plot(min_from_start, energy_real, color='steelblue', label='$E_{real}$', linewidth=linewidth)
    ax1.plot(min_from_start, energy_calc, color='orange', label='$E_{calc}$', linewidth=linewidth)
    ax1.set_xlabel('Match time (min)')
    ax1.set_ylabel('Cumulative E (J/kg)')
    ax1.set_ylim(0, 55000)

    ax2.plot(min_from_start, gd_arr, color='gray', label='GD', linewidth=linewidth)
    ax2.set_xlabel('Match time (min)')
    ax2.set_ylabel('GD')
    ax2.set_title(f'GD margin')
    # ax2.set_facecolor('xkcd:white')
    ax2.set_yticks(np.arange(-2, 3))

    team_state_acronym = pge_state.capitalize()[0]
    ax3.plot(
        min_from_start,
        correction_coef_arr,
        color='lightblue',
        label=f'$G_{team_state_acronym}$',
        linewidth=linewidth
    )
    ax3.set_xlabel('Match time (min)')
    ax3.set_ylabel(f'$G_{team_state_acronym}$')
    ax3.set_title(f'G when PGE={team_state_acronym.lower()}')
    ax3.set_ylim((0.4, 1.6))
    # ax3.set_facecolor('xkcd:white')

    fig.legend(loc='center right')
    return fig
