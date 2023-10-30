from typing import TypedDict, List

import numpy as np

PlayerMatchMeasurement = TypedDict(
    'PlayerMatchMeasurement', {
        'min_from_start': np.array,
        'duration_min': np.array,
        'energy_per_min': np.array,
        'gd': np.array,
        'transition_gd': np.array,
        'last_gd': np.array,
        'betting_position': np.array,
        'betting_position_reduced': np.array,
        'real_values': np.array,
        'calculated_values': np.array,
        'num_minutes': int,
        'game_label': str,
    })

GDEnergyDict = TypedDict(
    'GDEnergyDict', {
        '-2': float,
        '-1': float,
        '0': float,
        '1': float,
        '2': float,
    }
)

BaselineEnergyDict = TypedDict(
    'BaselineEnergyDict', {
        'favorite': GDEnergyDict,
        'non-favorite': GDEnergyDict
    }
)
GDPowerDict = TypedDict(
    'GDPowerDict', {
        '-2': float,
        '-1': float,
        '0': float,
        '1': float,
        '2': float,
    }
)
PlayerMeasurement = TypedDict(
    'PlayerMeasurement',
    {
        'values': List[PlayerMatchMeasurement],
        'total_minutes': int,
        'num_matches': int,
        'base_energy_median': BaselineEnergyDict,
        'base_energy_mean': BaselineEnergyDict,
    }
)


TransitionStateDict = TypedDict(
    'TransitionStateDict', {
        '-2_-1': float,
        '-1_-2': float,
        '-1_0': float,
        '0_-1': float,
        '0_1': float,
        '1_0': float,
        '1_2': float,
        '2_1': float,
    }
)

RadarColors = TypedDict(
    'RadarColors', {
        'facecolor': str,
        'edgecolor': str,
    }
)

MplRadarPlotColors = TypedDict(
    'MplRadarPlotColors', {
        'rings_inner': RadarColors,
        'radar': RadarColors,
    }
)
