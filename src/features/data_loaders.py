import os
import pandas as pd
from settings import PROCESSED_DATA_DIR
from src.features.optimisation.data_helpers import DataHolder

MAX_REALISTIC_POWER = 2000


def load_optimisation_data(is_dummy=False) -> DataHolder:
    filename = 'data_df.csv'
    if is_dummy:
        filename = 'data_df_one_player.csv'
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, filename))
    df.loc[df.energy_per_min > MAX_REALISTIC_POWER, 'energy_per_min'] = MAX_REALISTIC_POWER
    df.loc[:, 'athlete_id'] = df.athlete.str.extract('(\d+)')
    df.loc[:, 'athlete_id'] = df.athlete_id.astype(int)
    data_holder_obj = DataHolder(df=df)
    return data_holder_obj


def load_df_data(is_dummy=False) -> pd.DataFrame:
    filename = 'data_df.csv'
    if is_dummy:
        filename = 'data_df_one_player.csv'
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, filename))
    df.loc[df.energy_per_min > MAX_REALISTIC_POWER, 'energy_per_min'] = MAX_REALISTIC_POWER
    df.loc[:, 'athlete_id'] = df.athlete.str.extract('(\d+)')
    df.loc[:, 'athlete_id'] = df.athlete_id.astype(int)
    return df
