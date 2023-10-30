import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')

REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
PLAYER_GD_MINUTES_DIR = os.path.join(FIGURES_DIR, 'player_gd_minutes')
NELDER_MEAD_FIG_DIR = os.path.join(FIGURES_DIR, 'nelder-mead')
NELDER_MEAD_REPORTS_DIR = os.path.join(REPORTS_DIR, 'nelder-mead')
PSO_FIG_DIR = os.path.join(FIGURES_DIR, 'pso')
PSO_REPORTS_DIR = os.path.join(REPORTS_DIR, 'pso')

USE_DUMMY = False
ETA_SIGN = u'\u03B7'
