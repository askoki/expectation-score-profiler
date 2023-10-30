from src.features.typing import RadarColors


def create_mpl_radar_color(facecolor: str, edgecolor: str) -> RadarColors:
    return {
        'facecolor': facecolor,
        'edgecolor': edgecolor,
    }
