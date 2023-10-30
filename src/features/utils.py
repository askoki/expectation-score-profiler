from datetime import datetime
from typing import Tuple


def log(message: str) -> None:
    timestamp = datetime.now().strftime('%H:%M:%S %d/%m/%y')
    print(f'{timestamp}: {message}')


def get_duration_hour_min_sec(start: float, end: float) -> Tuple[float, float, float]:
    hours = (end - start) // 3600
    minutes = ((end - start) - hours * 3600) // 60
    seconds = int((end - start) - hours * 3600 - minutes * 60)
    return hours, minutes, seconds