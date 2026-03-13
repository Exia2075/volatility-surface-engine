DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_DIVIDEND_YIELD = 0.0

DEFAULT_OPTION_TYPE = "call"

MIN_T_DAYS = 7 # ignore options expiring in less than 1 week
MAX_T_DAYS = 730 # ignore options expiring in more than 2 yrs

REMOVE_ILLIQUID = True
MIN_VOLUME = 0
MIN_OPEN_INTEREST = 0

IV_TOLERANCE = 1e-6
IV_MAX_ITER_NR = 100
IV_MAX_ITER_BIS = 200
SIGMA_LOW = 1e-6
SIGMA_HIGH = 10.0

DEFAULT_AXIS_MODE = "moneyness"
DEFAULT_GRID_SIZE = 50

DEFAULT_COLORMAP = "RdYlGn_r" # red = high vol, green = low vol
DEFAULT_SAVE_PATH = "vol_surface.png"

DB_PATH = "vol_surface_history.db" # for future storage
