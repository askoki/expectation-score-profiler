import os
import matplotlib.pyplot as plt

from settings import USE_DUMMY, PLAYER_GD_MINUTES_DIR
from src.features.data_loaders import load_df_data

from src.features.file_helpers import create_dir
from src.visualization.helpers.matplotlib_style import load_plt_style

df = load_df_data(is_dummy=USE_DUMMY)

players_list = df.sort_values('athlete_id').athlete.unique()
players_count = len(players_list)

for i, player_name in enumerate(players_list):
    print(f'Processing player: {player_name} {i + 1}/{players_count}')
    create_dir(PLAYER_GD_MINUTES_DIR)

    p_df = df[df.athlete == player_name]
    gd_states = [-2, -1, 0, 1, 2]
    minutes_per_gd = []
    for gd in gd_states:
        gd_df = p_df[p_df.gd == gd]
        minutes_per_gd.append(gd_df.duration_min.sum())

    load_plt_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plt.rcParams['axes.unicode_minus'] = True
    plt.title('Minutes per GD')
    ax.bar(gd_states, minutes_per_gd)
    ax.set_ylabel('Minutes cumulative')
    ax.set_xlabel('GD')
    fig.savefig(os.path.join(PLAYER_GD_MINUTES_DIR, f'{player_name}_player_score_minute.png'), dpi=300)
    plt.close()
