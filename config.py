# Threshold and selection settings for Deribit monitor.
#
# Adjust values here instead of editing the main script.

# Rolling windows (seconds)
WINDOW_SEC = 300          # for z-scores, IV/skew jumps, OI accel
RET_WINDOW_SEC = 60       # short window for fast price moves

# Price / IV / liquidity alerts
PRICE_Z = 3.0             # mark-price z-score
IV_JUMP = 0.05            # absolute IV jump (5 vols)
RET_PCT = 0.015           # 1.5% move within RET_WINDOW_SEC
# Liquidity stress alert threshold (set to None to disable)
SPREAD_WIDEN = None

# Skew (25-delta risk reversal) signals
DELTA_TARGET = 0.25
DELTA_BAND = 0.10
SKEW_INV_ALERT = -0.10    # skew below -10 vols (puts rich)
SKEW_JUMP = 0.05          # |Δskew| ≥ 5 vols in WINDOW_SEC

# Leverage / perp OI acceleration
OI_ACCEL_PCT = 0.05       # +5% OI in WINDOW_SEC

# Instrument selection
ATM_EXPIRIES = 4          # monitor next ~week+ (first 4 expiries)
ATM_STRIKES_PER_EXPIRY = 2
OTM_EXPIRIES = 4          # include farther-dated OTM tails
OTM_PER_SIDE = 2
OTM_THRESHOLD = 0.20      # ±20% from spot counts as far OTM

# Risk-duty clustering (reduce noisy per-contract alerts)
CLUSTER_WINDOW_SEC = 300   # lookback window for clustering alerts
CLUSTER_THRESHOLD = 5      # send one summary when >= this many alerts in window for same expiry

# Composite gating: when True, send a single composite alert only if ALL key metrics are hit.
# Metrics reused: PRICE_Z, RET_PCT, IV_JUMP, and SPREAD_WIDEN (if set).
COMPOSITE_ALERT_ONLY = True

# Large aggressive trade alerts (options)
LARGE_TRADE_NOTIONAL = 500000000     # USD underlying notional threshold (index_price * amount)
LARGE_TRADE_PREMIUM = 500000      # USD premium threshold (price * amount)

# Snapshot reporting interval
SNAPSHOT_INTERVAL_SEC = 86400       # Time between periodic snapshots (5 minutes)

# DVOL (Volatility Index) monitor settings
DVOL_WINDOW_SEC = 3600             # 1 hour rolling window for DVOL change detection
# DVOL_RISE_THRESHOLD = 0.05         # 5% DVOL rise in 1h (IV pulse)
# PRICE_DROP_THRESHOLD = -0.025      # -2.5% price drop in 1h (bearish filter)
DVOL_RISE_THRESHOLD = 0.05       
PRICE_DROP_THRESHOLD = -0.025

DVOL_SNAPSHOT_INTERVAL_SEC = 14400  # 4 hours snapshot interval (4 * 3600)

# Sell-put candidate selection (used on DVOL alert)
SELL_PUT_DTE_MIN = 23
SELL_PUT_DTE_MAX = 35
SELL_PUT_DELTA_TARGET = 0.15
SELL_PUT_DELTA_BAND = 0.05
SELL_PUT_SCORE = "premium_per_delta"
SELL_PUT_TOPN = 3
SELL_PUT_MAX_CONCURRENCY = 20
