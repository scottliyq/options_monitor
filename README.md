# options_monitor

Monitors Deribit BTC/ETH options for fast moves, IV/skew anomalies, and large aggressive trades. Ships three independent scripts:

- `deribit_vol_monitor.py`: real-time options/perp monitoring with alerts on price z-score, IV jumps, fast moves, skew shifts, and (optionally) liquidity stress.
- `large_trade_monitor.py`: listens to Deribit option trade feed and alerts on large taker buys by USD notional or premium.
- `dvol_monitor.py`: monitors DVOL (Deribit Volatility Index) for IV pulse events combined with price drops.

## Requirements

- Python 3.9+ (tested with 3.12 env)
- Install deps: `pip install -r requirements.txt`
- Deribit API key for option trades feed (public/auth): set `DERIBIT_CLIENT_ID` / `DERIBIT_CLIENT_SECRET` in `.env`. Use `DERIBIT_TESTNET=1` to target testnet.
- Telegram bot for alerts: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. Webhook alerts also supported via `ALERT_WEBHOOK`.
- Pushover emergency alerts (DVOL alerts only): `PUSHOVER_USER_KEY` and `PUSHOVER_API_TOKEN` in `.env`.

## Configuration (`config.py`)

Key knobs (defaults shown):

- Rolling/alert windows: `WINDOW_SEC=120`, `RET_WINDOW_SEC=30`
- Price/IV/returns: `PRICE_Z=3.0`, `IV_JUMP=0.05`, `RET_PCT=0.015`
- Skew: `DELTA_TARGET=0.25`, `DELTA_BAND=0.10`, `SKEW_INV_ALERT=-0.10`, `SKEW_JUMP=0.05`
- Liquidity: `SPREAD_WIDEN=None` (set a float like `0.1` to re-enable)
- Instrument selection: `ATM_EXPIRIES=4`, `OTM_EXPIRIES=4`, `ATM_STRIKES_PER_EXPIRY=2`, `OTM_PER_SIDE=2`, `OTM_THRESHOLD=0.20`
- Clustered duty alerts: `CLUSTER_WINDOW_SEC=120`, `CLUSTER_THRESHOLD=3`
- Large trades (options): `LARGE_TRADE_NOTIONAL` (USD notional via index * size), `LARGE_TRADE_PREMIUM` (USD premium via price * size * index)
- DVOL monitor: `DVOL_WINDOW_SEC=3600` (1h window), `DVOL_RISE_THRESHOLD=0.05` (5% DVOL rise), `PRICE_DROP_THRESHOLD=-0.025` (-2.5% price drop)

## Environment (`.env`)

```
DERIBIT_CLIENT_ID=xxx
DERIBIT_CLIENT_SECRET=yyy
TELEGRAM_BOT_TOKEN=zzz
TELEGRAM_CHAT_ID=123456
PUSHOVER_USER_KEY=your_user_key
PUSHOVER_API_TOKEN=your_app_token
# Optional
ALERT_WEBHOOK=https://...
DERIBIT_TESTNET=1   # omit for mainnet
LOG_LEVEL=INFO
PUSHOVER_RETRY_SEC=60   # optional, emergency alert repeat interval
PUSHOVER_EXPIRE_SEC=3600 # optional, emergency alert max duration
```

## Running

1) Install deps:
```bash
pip install -r requirements.txt
```

2) Volatility/IV monitor:
```bash
python deribit_vol_monitor.py
```
- Sends alerts to Telegram/Webhook (if configured).
- Logs an immediate snapshot, then every 30 minutes.

3) Large options trade monitor:
```bash
python large_trade_monitor.py
```
- Authenticates with Deribit, subscribes to `trades.option.BTC/ETH`.
- Alerts when `notional >= LARGE_TRADE_NOTIONAL` **or** `premium >= LARGE_TRADE_PREMIUM`.

4) DVOL (Volatility Index) monitor:
```bash
python dvol_monitor.py
```
- Monitors BTC and ETH DVOL indices for IV pulse events.
- Alerts when **both conditions** are met simultaneously:
  - DVOL 1h rise ≥ 5% (configurable via `DVOL_RISE_THRESHOLD`)
  - Price 1h drop ≤ -2.5% (configurable via `PRICE_DROP_THRESHOLD`)
- Sends highlighted Telegram alerts with detailed metrics.
- Sends Pushover emergency alerts (`priority=2`) together with Telegram when an alert triggers.
- 10-minute cooldown between alerts per currency to avoid spam.

## Notes

- Set `DERIBIT_TESTNET=1` plus a testnet API key if you want to avoid live trading data.
- Both scripts run until interrupted; they auto-reconnect with backoff.
- Tune thresholds in `config.py` to match your duty/alert sensitivity.
