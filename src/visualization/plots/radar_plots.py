import matplotlib.pyplot as plt
from typing import List, Tuple
from mplsoccer import Radar, grid
from src.features.typing import MplRadarPlotColors

MatplotlibSubplots = Tuple[plt.Figure, plt.Axes]


def plot_mpl_radar(values: List[float], labels: List[str], bound_low: List[float],
                   bound_high: List[float], colors: MplRadarPlotColors, round_int=list or bool) -> MatplotlibSubplots or None:
    round_int = round_int if type(round_int) == list else [True] * len(labels)
    radar = Radar(
        labels,
        bound_low,
        bound_high,
        round_int=round_int,
        num_rings=4,
        ring_width=1,
        center_circle_radius=1
    )
    fig, axs = grid(
        figheight=4,
        grid_height=0.915,
        title_height=0.06,
        endnote_height=0.025,
        title_space=0,
        endnote_space=0,
        grid_key='radar',
        axis=False
    )
    radar.setup_axis(ax=axs['radar'])
    rings_inner = radar.draw_circles(
        ax=axs['radar'], facecolor=colors['rings_inner']['facecolor'], edgecolor=colors['rings_inner']['edgecolor']
    )
    radar3, vertices3 = radar.draw_radar_solid(
        values, ax=axs['radar'],
        kwargs={
            'facecolor': colors['radar']['facecolor'],
            'alpha': 0.6,
            'edgecolor': colors['radar']['edgecolor'],
            'lw': 3
        }
    )
    axs['radar'].scatter(
        vertices3[:, 0], vertices3[:, 1],
        c=colors['radar']['facecolor'], edgecolors=colors['radar']['edgecolor'], marker='o', s=50, zorder=2
    )
    range_labels = radar.draw_range_labels(ax=axs['radar'], fontsize=6)
    param_labels = radar.draw_param_labels(ax=axs['radar'], fontsize=10)
    lines = radar.spoke(ax=axs['radar'], color='#a6a4a1', linestyle='--', zorder=2)
    return fig, axs['radar']


ComparatorLabels = List[str]
ComparatorListValues = List[List[float]]
ComparatorListLegendLabels = List[List[str]]
