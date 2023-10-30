import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from settings import INTERIM_DATA_DIR, FIGURES_DIR
from src.visualization.constants import PGE_STATES, REDUCED_PGE_STATES
from src.visualization.helpers.processing import add_labels_to_cm

fcu_9state_df = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'fcu_9states_all_players.csv'))
fnf_9state_df = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'fnf_9states_all_players.csv'))
fcu_4state_df = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'fcu_4state_all_players.csv'))
fnf_4state_df = pd.read_csv(os.path.join(INTERIM_DATA_DIR, 'fnf_4state_all_players.csv'))

fcu_9s_matrix = fcu_9state_df.drop(columns=['state']).to_numpy()
fnf_9s_matrix = fnf_9state_df.drop(columns=['state']).to_numpy()
fcu_4s_matrix = fcu_4state_df.drop(columns=['state']).to_numpy()
fnf_4s_matrix = fnf_4state_df.drop(columns=['state']).to_numpy()

# plot these four matrices on the same chart
fig, axs = plt.subplots(
    nrows=2, ncols=2, figsize=(4, 8), gridspec_kw={'height_ratios': [1.8, 1]},

)

min_val = 0
max_val = 19
axs[0][0].imshow(fcu_9s_matrix, cmap='Blues', interpolation='nearest', vmin=min_val, vmax=max_val)
axs[0][1].imshow(fnf_9s_matrix, cmap='Blues', interpolation='nearest', vmin=min_val, vmax=max_val)
axs[1][0].imshow(fcu_4s_matrix, cmap='Blues', interpolation='nearest', vmin=min_val, vmax=max_val)
axs[1][1].imshow(fnf_4s_matrix, cmap='Blues', interpolation='nearest', vmin=min_val, vmax=max_val)

axs[0][0].set_xticks(np.arange(3), PGE_STATES, rotation=45)
axs[1][0].set_xticks(np.arange(3), PGE_STATES, rotation=45)
axs[0][1].set_xticks(np.arange(2), REDUCED_PGE_STATES, rotation=45)
axs[1][1].set_xticks(np.arange(2), REDUCED_PGE_STATES, rotation=45)

axs[0][1].set_yticks(np.arange(len(fcu_9state_df.state)), fcu_9state_df.state)
axs[1][1].set_yticks(np.arange(len(fcu_4state_df.state)), fcu_4state_df.state)
axs[0][0].set_yticks([])
axs[1][0].set_yticks([])
axs[0][1].set_xticks([])
axs[0][0].set_xticks([])

axs[0][0].set_ylabel('GDC')
axs[1][0].set_ylabel('GD')

add_labels_to_cm(axs[0][0], fcu_9s_matrix)
add_labels_to_cm(axs[0][1], fnf_9s_matrix)
add_labels_to_cm(axs[1][0], fcu_4s_matrix)
add_labels_to_cm(axs[1][1], fnf_4s_matrix)

plt.subplots_adjust(hspace=0.05)
plt.savefig(os.path.join(FIGURES_DIR, 'all_players_data_stats.png'), dpi=300)