import os
import matplotlib.pyplot as plt
import numpy as np

from settings import INTERIM_DATA_DIR
from src.features.data_loaders import load_df_data

from src.features.file_helpers import create_dir
from src.visualization.constants import GD_STATES

df = load_df_data(is_dummy=False)

players_list = df.sort_values('athlete_id').athlete.unique()
players_count = len(players_list)

max_min_per_gd = df.groupby(['athlete', 'gd']).agg({'duration_min': 'sum'}).duration_min.max()
fig, axs = plt.subplots(
    nrows=players_count, ncols=len(GD_STATES) + 1, sharey='row',
    figsize=(4, 1 * players_count)
)
energy_threshold = 1000
for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    save_path = os.path.join(INTERIM_DATA_DIR, player_name)
    create_dir(save_path)

    p_axs = axs[i]
    p_df = df[(df.athlete == player_name) & (df.energy_per_min <= energy_threshold)]
    minutes_per_gd = []
    for ax, gd_state in zip(p_axs[:-1], GD_STATES):
        gd_df = p_df[p_df.gd == gd_state]
        ax.hist(gd_df.energy_per_min, density=True)

        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel(f'GD{gd_state}', fontsize=6, fontweight='bold')
        ax.xaxis.set_tick_params(labeltop=True)
        minutes_per_gd.append(gd_df.duration_min.sum())
    minutes_ax = p_axs[-1]
    minutes_ax.spines['top'].set_visible(False)
    minutes_ax.spines['left'].set_visible(False)
    minutes_ax.spines['bottom'].set_visible(False)
    minutes_ax2 = minutes_ax.twinx()
    minutes_ax2.set_ylim(-0.05, 1.05)
    minutes_ax2.set_yticks([])
    minutes_ax2.set_xticks([])
    minutes_ax.set_xticks([])
    minutes_ax2.spines['top'].set_visible(False)
    minutes_ax2.spines['right'].set_visible(False)

    minutes_ax2.bar(GD_STATES, np.array(minutes_per_gd) / max_min_per_gd, color='tomato', hatch='///')

    p_axs[-1].spines['right'].set_visible(False)
    p_axs[-1].set_xlabel(f'Min per GD', fontsize=6, fontweight='bold')
    p_axs[0].set_ylabel(player_name, fontsize=6, fontweight='bold')

plt.tight_layout()
plt.xticks(fontsize=4)
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()
# plt.savefig(os.path.join(FIGURES_DIR, 'all_players_energy_distribution.png'), dpi=300)
