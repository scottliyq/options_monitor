#!/usr/bin/env python3
"""
Deribit BTC/ETH option monitor.

Watches near-dated, near-ATM options via Deribit's public API and raises alerts
when fast price/IV changes or liquidity stress likely precede large moves.

Run:
  python deribit_vol_monitor.py

Environment:
  ALERT_WEBHOOK   Optional Slack/Teams-compatible webhook URL for alerts.
  DERIBIT_TESTNET Set to "1" to use the Deribit testnet.
"""

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from statistics import mean, pstdev
from typing import Deque, Dict, Iterable, List, Tuple

import aiohttp
import websockets

import config
import logging

BASE_URL = (
    "https://test.deribit.com/api/v2"
    if os.getenv("DERIBIT_TESTNET")
    else "https://www.deribit.com/api/v2"
)
WS_URL = (
    "wss://test.deribit.com/ws/api/v2"
    if os.getenv("DERIBIT_TESTNET")
    else "wss://www.deribit.com/ws/api/v2"
)


class RollingWindow:
    """Fixed-duration rolling window of (timestamp, value) points."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.data: Deque[Tuple[float, float]] = deque()

    def add(self, ts: float, value: float) -> None:
        self.data.append((ts, value))
        cutoff = ts - self.seconds
        while self.data and self.data[0][0] < cutoff:
            self.data.popleft()

    def stats(self) -> Tuple[float, float]:
        if len(self.data) < 2:
            return (float("nan"), float("nan"))
        vals = [v for _, v in self.data]
        return mean(vals), pstdev(vals)

    def pct_move(self) -> float:
        if len(self.data) < 2:
            return 0.0
        start = self.data[0][1]
        end = self.data[-1][1]
        if start == 0:
            return 0.0
        return (end - start) / start


async def http_get(session: aiohttp.ClientSession, path: str, params: Dict) -> Dict:
    url = f"{BASE_URL}{path}"
    async with session.get(url, params=params, timeout=10) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_index_price(session: aiohttp.ClientSession, currency: str) -> float:
    res = await http_get(session, "/public/get_index_price", {"index_name": f"{currency.lower()}_usd"})
    return res["result"]["index_price"]


async def fetch_instruments(session: aiohttp.ClientSession, currency: str) -> List[Dict]:
    res = await http_get(
        session,
        "/public/get_instruments",
        {
            "currency": currency,
            "kind": "option",
            # Deribit expects "true"/"false" strings; aiohttp/yarl rejects bare bools
            "expired": "false",
        },
    )
    return res["result"]


def _parse_option(ins: Dict) -> Tuple[float, str]:
    """
    Extract strike and right ("C" / "P") from Deribit instrument.
    Instrument example: BTC-28JUN24-60000-C
    """
    name: str = ins["instrument_name"]
    parts = name.split("-")
    if len(parts) < 4:
        return 0.0, ""
    strike = float(parts[-2])
    right = parts[-1]
    return strike, right


def pick_atm_instruments(
    instruments: List[Dict],
    index_price: float,
    expiries: int = config.ATM_EXPIRIES,
    strikes_per_expiry: int = config.ATM_STRIKES_PER_EXPIRY,
) -> List[str]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for ins in instruments:
        grouped[ins["expiration_timestamp"]].append(ins)
    chosen: List[str] = []
    for expiry in sorted(grouped.keys())[:expiries]:
        bucket = grouped[expiry]
        bucket.sort(key=lambda x: abs(x["strike"] - index_price))
        for ins in bucket[: strikes_per_expiry * 2]:
            chosen.append(ins["instrument_name"])
    return chosen


def pick_far_otm_instruments(
    instruments: List[Dict],
    index_price: float,
    expiries: int = config.OTM_EXPIRIES,
    per_side: int = config.OTM_PER_SIDE,
    otm_threshold: float = config.OTM_THRESHOLD,
) -> List[str]:
    """
    Pick far OTM calls (strike >= (1+thr)*spot) and puts (strike <= (1-thr)*spot)
    for the nearest expiries.
    """
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for ins in instruments:
        grouped[ins["expiration_timestamp"]].append(ins)

    chosen: List[str] = []
    upper = index_price * (1 + otm_threshold)
    lower = index_price * (1 - otm_threshold)
    for expiry in sorted(grouped.keys())[:expiries]:
        bucket = grouped[expiry]
        calls, puts = [], []
        for ins in bucket:
            strike, right = _parse_option(ins)
            if right == "C" and strike >= upper:
                calls.append(ins)
            elif right == "P" and strike <= lower:
                puts.append(ins)
        calls.sort(key=lambda x: x["strike"])
        puts.sort(key=lambda x: x["strike"], reverse=True)
        for ins in calls[:per_side]:
            chosen.append(ins["instrument_name"])
        for ins in puts[:per_side]:
            chosen.append(ins["instrument_name"])
    return chosen


def format_alert(title: str, body: str, extra: Dict) -> str:
    bits = [title, body] + [f"{k}={v}" for k, v in extra.items()]
    return " | ".join(bits)


async def send_webhook(text: str) -> None:
    url = os.getenv("ALERT_WEBHOOK")
    if not url:
        return
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(url, json={"text": text}, timeout=5)
        except Exception:
            pass  # avoid crashing monitor on webhook errors


async def build_watch_list(session: aiohttp.ClientSession) -> List[str]:
    watch: List[str] = []
    for currency in ("BTC", "ETH"):
        idx = await fetch_index_price(session, currency)
        instruments = await fetch_instruments(session, currency)
        watch += pick_atm_instruments(instruments, idx)
        watch += pick_far_otm_instruments(instruments, idx)
    return watch


async def subscribe(ws, channels: Iterable[str]) -> None:
    await ws.send(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "public/subscribe",
                "params": {"channels": list(channels)},
            }
        )
    )


async def monitor() -> None:
    logger = logging.getLogger("deribit_monitor")
    backoff = 1
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                watch_list = await build_watch_list(session)
            channels = [f"ticker.{ins}.agg2" for ins in watch_list] + [
                "deribit_price_index.btc_usd",
                "deribit_price_index.eth_usd",
                "ticker.BTC-PERPETUAL.agg2",
                "ticker.ETH-PERPETUAL.agg2",
            ]
            logger.info("Subscribing to %d options...", len(watch_list))
            for idx, ins in enumerate(watch_list, 1):
                logger.info("  [%d/%d] %s", idx, len(watch_list), ins)
            async with websockets.connect(WS_URL, ping_interval=15, ping_timeout=10) as ws:
                await subscribe(ws, channels)
                backoff = 1
                prices: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(config.WINDOW_SEC))
                ivs: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(config.WINDOW_SEC))
                spread: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(config.WINDOW_SEC))
                recent: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(config.RET_WINDOW_SEC))
                skew_series: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(config.WINDOW_SEC))  # key: BTC or ETH skew time series
                latest_call_iv: Dict[str, float] = {}
                latest_put_iv: Dict[str, float] = {}
                oi_series: Dict[str, RollingWindow] = defaultdict(lambda: RollingWindow(config.WINDOW_SEC))    # key: BTC/ETH perp

                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("method") != "subscription":
                        continue
                    channel = msg["params"]["channel"]
                    data = msg["params"]["data"]
                    ts = data.get("timestamp", time.time())

                    if channel.startswith("ticker."):
                        ins = data["instrument_name"]
                        mark = float(data["mark_price"])
                        mark_iv = float(data.get("mark_iv", 0.0))
                        bid = float(data.get("best_bid_price") or 0)
                        ask = float(data.get("best_ask_price") or 0)
                        underlying = float(
                            data.get("underlying_price")
                            or data.get("underlying_index_price")
                            or mark
                        )
                        strike, right = _parse_option({"instrument_name": ins})

                        # Futures / perp leverage monitor
                        if ins.endswith("-PERPETUAL"):
                            currency = ins.split("-")[0]
                            oi = float(data.get("open_interest") or 0.0)
                            oi_series[currency].add(ts, oi)
                            if abs(oi_series[currency].pct_move()) >= config.OI_ACCEL_PCT:
                                text = format_alert(
                                    "Perp OI accelerating",
                                    f"{ins} OI +{oi_series[currency].pct_move():.1%} in {config.WINDOW_SEC}s",
                                    {"oi": oi},
                                )
                                logger.warning(text)
                                await send_webhook(text)
                            continue

                        prices[ins].add(ts, mark)
                        ivs[ins].add(ts, mark_iv)
                        if bid and ask and mark:
                            spread[ins].add(ts, (ask - bid) / mark)
                        recent[ins].add(ts, mark)

                        mean_p, std_p = prices[ins].stats()
                        z = (mark - mean_p) / std_p if std_p and std_p > 0 else 0.0
                        iv_mean, iv_std = ivs[ins].stats()
                        iv_jump = abs(mark_iv - iv_mean) if not (iv_std != iv_std) else 0.0
                        move = abs(recent[ins].pct_move())
                        spr = spread[ins].data[-1][1] if spread[ins].data else 0.0

                        if z >= config.PRICE_Z:
                            text = format_alert(
                                "Price z-score spike",
                                f"{ins} z={z:.1f}",
                                {"mark": mark, "mean": f"{mean_p:.2f}", "std": f"{std_p:.2f}"},
                            )
                            logger.warning(text)
                            await send_webhook(text)
                        is_far_otm = False
                        if underlying > 0 and strike > 0 and right:
                            is_far_otm = (
                                (right == "C" and strike >= underlying * 1.20)
                                or (right == "P" and strike <= underlying * 0.80)
                            )

                        if iv_jump >= config.IV_JUMP:
                            label = "Far OTM IV jump" if is_far_otm else "IV jump"
                            text = format_alert(
                                label,
                                f"{ins} ΔIV={iv_jump:.2%}",
                                {
                                    "mark_iv": f"{mark_iv:.2%}",
                                    "iv_mean": f"{iv_mean:.2%}",
                                    "strike": strike,
                                    "under": f"{underlying:.2f}",
                                },
                            )
                            logger.warning(text)
                            await send_webhook(text)
                        if move >= config.RET_PCT:
                            text = format_alert(
                                "Fast move",
                                f"{ins} {move:.2%} in {config.RET_WINDOW_SEC}s",
                                {"mark": mark},
                            )
                            logger.warning(text)
                            await send_webhook(text)
                        if config.SPREAD_WIDEN is not None and spr >= config.SPREAD_WIDEN:
                            text = format_alert(
                                "Liquidity stress",
                                f"{ins} spread {spr:.1%}",
                                {"bid": bid, "ask": ask, "mark": mark},
                            )
                            logger.warning(text)
                            await send_webhook(text)

                        # Skew monitoring (25-delta RR proxy)
                        if right:
                            currency = ins.split("-")[0]
                            delta_raw = data.get("greeks", {}).get("delta") if isinstance(data.get("greeks"), dict) else data.get("delta")
                            try:
                                delta = float(delta_raw or 0.0)
                            except Exception:
                                delta = 0.0
                            if right == "C" and abs(delta - config.DELTA_TARGET) <= config.DELTA_BAND:
                                latest_call_iv[currency] = mark_iv
                            elif right == "P" and abs(delta + config.DELTA_TARGET) <= config.DELTA_BAND:
                                latest_put_iv[currency] = mark_iv

                            if currency in latest_call_iv and currency in latest_put_iv:
                                call_iv = latest_call_iv[currency]
                                put_iv = latest_put_iv[currency]
                                if not (call_iv != call_iv or put_iv != put_iv):
                                    skew = call_iv - put_iv
                                    skew_series[currency].add(ts, skew)
                                    skew_mean, skew_std = skew_series[currency].stats()
                                    skew_change = skew_series[currency].pct_move()

                                    if skew <= config.SKEW_INV_ALERT:
                                        text = format_alert(
                                            "Skew inverted (puts rich)",
                                            f"{currency} skew={skew:.2%}",
                                            {"call_iv": f"{call_iv:.2%}", "put_iv": f"{put_iv:.2%}"},
                                        )
                                        logger.warning(text)
                                        await send_webhook(text)
                                    elif abs(skew_change) >= config.SKEW_JUMP:
                                        text = format_alert(
                                            "Skew jump",
                                            f"{currency} Δskew={skew_change:.2%} in {config.WINDOW_SEC}s",
                                            {"skew": f"{skew:.2%}", "mean": f"{skew_mean:.2%}"},
                                        )
                                        logger.warning(text)
                                        await send_webhook(text)

                    elif channel.startswith("deribit_price_index."):
                        # could add underlying move-based alerts if desired
                        pass
        except Exception as exc:
            logger.exception("Monitor error: %s; reconnecting in %ss", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        asyncio.run(monitor())
    except KeyboardInterrupt:
        logging.getLogger("deribit_monitor").info("Stopped by user.")
