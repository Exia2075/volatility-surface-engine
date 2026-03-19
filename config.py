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

RBF_SMOOTHING = 0.001
IV_CLIP_LOW = 0.05
IV_CLIP_HIGH = 2.0

DEFAULT_COLORMAP = "RdYlGn_r" # red = high vol, green = low vol
DEFAULT_SAVE_PATH = "vol_surface.png"

PLOT_DPI = 150
PLOT_FIGSIZE = (14, 9)

API_RATE_LIMIT_DELAY = 0.1 # Seconds between API calls (rate limiting)
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1.0 # Seconds to wait before retry

DB_PATH = "vol_surface_history.db" # for future storage
DB_BATCH_SIZE = 100 # Number of surfaces to batch before commit