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
LARGE_TRADE_NOTIONAL = 5000000     # USD underlying notional threshold (index_price * amount)
LARGE_TRADE_PREMIUM = 500000      # USD premium threshold (price * amount)

# Snapshot reporting interval
SNAPSHOT_INTERVAL_SEC = 3600       # Time between periodic snapshots (5 minutes)
