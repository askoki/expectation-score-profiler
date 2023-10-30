import os
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from settings import FIGURES_DIR
from src.features.optimisation.processing import calc_fatigue_effect, OptInfo
from src.visualization.helpers.processing import convert_gd_to_step_fill_function

P_fav = {
    -2: 400,
    -1: 550,
    0: 500,
    1: 600,
    2: 350
}

P_non = {
    -2: 300,
    -1: 500,
    0: 400,
    1: 350,
    2: 360
}

stamina = 0.7
input_vector = [
    *list(P_fav.values()),
    *list(P_non.values()),
    stamina
]
opt_info = OptInfo(input_vector)

GoalInfo = namedtuple('GoalInfo', ['start', 'end', 'goal_diff'])
goal_info_list = [
    GoalInfo(0, 25, 0),
    GoalInfo(25, 60, 1),
    GoalInfo(60, 70, 0),
    GoalInfo(70, 90, -1)
]

goal_difference_list = []
playing_min = list(range(0, 91, 5))
for m in playing_min:
    for g_info in goal_info_list:
        if g_info.start <= m < g_info.end:
            goal_difference_list.append(g_info.goal_diff)
            break

power_array = [P_fav[gd] for gd in goal_difference_list]
power_array = np.array(power_array)

x = np.arange(0, 91)

energy_per_min = []
DURATION_MIN = 5
for p in power_array:
    energy_per_min.extend([p] * DURATION_MIN)
energy_per_min = np.array(energy_per_min)
energy_per_min = np.insert(energy_per_min, 0, 0, axis=0)
power_array = np.insert(power_array, 0, 0, axis=0)

fatigue_effect = calc_fatigue_effect(opt_info.stamina, x)
power_x = np.linspace(0, 90, 800)
fatigue_effect_power = calc_fatigue_effect(opt_info.stamina, power_x)


def get_power(t: int, g_info_list: list):
    for g_info in g_info_list:
        if g_info.start <= t <= g_info.end:
            return P_fav[g_info.goal_diff]


plt.rc('ytick', labelsize=6)
plt.rc('xtick', labelsize=6)
plt.rc('axes', labelsize=8)
plt.rc('legend', fontsize=8)

fig, ax = plt.subplots(figsize=(6, 4), nrows=3, sharex='col')
energy_cum = energy_per_min * fatigue_effect
ax[0].plot(
    x, energy_per_min.cumsum() / 1000, color='red',
    linewidth=3, label='Estimated energy'
)
ax[0].set_ylabel('Energy kJ/kg')
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].tick_params('x', labelbottom=False, length=0)
ax[0].set_ylim(0, 50)
ax[0].locator_params(axis='y', nbins=3)
ax[0].legend(bbox_to_anchor=(0.34, 1), loc='upper left', borderaxespad=0)
ax[0].tick_params(axis='y', colors='black')
ax[0].tick_params(axis='x', colors='black')
ax[0].text(
    x=45,
    y=15,
    s=f'$\eta={stamina}$',
    usetex=True,
    fontdict=dict(fontsize=6),
    ha='center',
    va='center'
)

goal_difference_list.insert(0, 0)
goal_difference_list = np.array(goal_difference_list)
ax[1].step(playing_min, goal_difference_list, where='pre', linewidth=2)
t, gd_y = convert_gd_to_step_fill_function(playing_min, goal_difference_list)
ax[1].fill_between(
    t, gd_y, where=(gd_y > 0), interpolate=True,
    color='forestgreen', alpha=0.5, step='pre'
)
ax[1].fill_between(
    t, gd_y, where=(gd_y < 0), interpolate=True,
    color='tomato', alpha=0.5, step='pre'
)
ax[1].set_ylabel('GD')
ax[1].spines['bottom'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

ax[1].tick_params('x', labelbottom=False, length=0)
ax[1].set_ylim((-2, 2))
# latex GD change equation
latex_code = r'$d_{s}(t) = \begin{cases}'
for gd_info in goal_info_list:
    latex_code += f'{gd_info.goal_diff}' + r'& \text{ for } t \in  [' + f'{gd_info.start},{gd_info.end}' + r'\rangle \text{,}\\'
latex_code += r'\end{cases}$'
ax[1].text(
    x=40,
    y=2.3,
    s=latex_code,
    usetex=True,
    fontdict=dict(fontsize=6),
    ha='center',
    va='center'
)

ax[2].set_ylabel('Power (W)')
ax[2].set_xlabel('Game time (min)')
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
power_y = [get_power(t, goal_info_list) for t in power_x]

ax[2].plot(power_x, power_y, color='darkblue', label='$B$', linewidth=3)
ax[2].plot(
    power_x, power_y * fatigue_effect_power, color='green',
    label='Model (η)', linewidth=3, alpha=0.8
)
ax[2].set_ylim(200, 800)
ax[2].locator_params(axis='y', nbins=2)

for gi in goal_info_list:
    ax[0].axvline(x=gi.start, linestyle='--', color='black')
    ax[2].axvline(x=gi.start, linestyle='--', color='black')
    ax[2].text(
        x=gi.start,
        y=850,
        s=f'{gi.start}',
        fontdict=dict(fontsize=6),
        ha='center',
        va='center'
    )
# input vector formula
latex_code = r'$P_{{s}}(d_{s}(t), f) =' + \
             r'\begin{cases}' + \
             f'{P_fav[-2]}' + r'& \text{ for } d(t)=-2 \text{,}\\ ' + \
             f'{P_fav[-1]}' + r'& \text{ for } d(t)=-1 \text{,}\\ ' + \
             f'{P_fav[0]}' + r'& \text{ for } d(t)=0 \text{,}\\ ' + \
             f'{P_fav[1]}' + r'& \text{ for } d(t)=1 \text{,}\\ ' + \
             f'{P_fav[2]}' + r'& \text{ for } d(t)=2\\' + \
             r'\end{cases}$'

params = {'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)
ax[2].text(
    x=42,
    y=780,
    s=latex_code,
    usetex=True,
    fontdict=dict(fontsize=6),
    ha='center',
    va='center'
)

ax[2].legend(bbox_to_anchor=(0.98, 1.1), loc='upper right', borderaxespad=0)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0)

fig.savefig(os.path.join(FIGURES_DIR, 'optimization_model_example_new.png'), dpi=300)
