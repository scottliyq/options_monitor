#!/usr/bin/env python3
"""
Monitor large aggressive buy trades on Deribit BTC/ETH perps and alert via Telegram.

Config:
  - LARGE_TRADE_NOTIONAL (USD) in config.py controls alert threshold.
Env:
  - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID for Telegram delivery.
  - DERIBIT_TESTNET=1 to use testnet endpoints.
"""

import asyncio
import json
import logging
import os
import time

import aiohttp
import websockets
from dotenv import load_dotenv

import config

load_dotenv()

BASE_WS = (
    "wss://test.deribit.com/ws/api/v2"
    if os.getenv("DERIBIT_TESTNET")
    else "wss://www.deribit.com/ws/api/v2"
)


async def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(url, json=payload, timeout=5)
        except Exception:
            pass


async def subscribe(ws, channels):
    await ws.send(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "public/subscribe",
                "params": {"channels": channels, "include_old": True},
            }
        )
    )


async def auth(ws) -> None:
    client_id = os.getenv("DERIBIT_CLIENT_ID")
    client_secret = os.getenv("DERIBIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("DERIBIT_CLIENT_ID/DERIBIT_CLIENT_SECRET not set in environment/.env")
    req = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "public/auth",
        "params": {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "read",
        },
    }
    await ws.send(json.dumps(req))
    resp_raw = await ws.recv()
    resp = json.loads(resp_raw)
    if resp.get("error"):
        raise RuntimeError(f"Auth error: {resp['error']}")
    return resp["result"].get("access_token")


def format_trade_alert(trade: dict, notional: float) -> str:
    ins = trade.get("instrument_name")
    price = float(trade.get("price") or 0.0)
    amt = float(trade.get("amount") or 0.0)
    side = trade.get("direction")
    idx = trade.get("_index_price")
    premium_usd = float(trade.get("_premium_usd") or 0.0)
    ts = trade.get("timestamp", int(time.time()))
    human_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts / 1000))
    price_str = f"{price:.5f}".rstrip("0").rstrip(".") if price else "0"
    size_str = f"{amt:.4f}".rstrip("0").rstrip(".") if amt else "0"
    return (
        f"Large taker {side} {ins}\n"
        f"Size: {size_str} @ {price_str}\n"
        f"Notional (est): ${notional:,.0f} (index {idx:.2f} USD)\n"
        f"Premium: ${premium_usd:,.2f} (price*size*index)\n"
        f"Time (UTC): {human_ts}"
    )


async def monitor() -> None:
    logger = logging.getLogger("large_trade_monitor")
    backoff = 1
    threshold = config.LARGE_TRADE_NOTIONAL
    # Option trades streams; raw for fullest feed
    channels = [
        "trades.option.BTC.raw",
        "trades.option.ETH.raw",
        "deribit_price_index.btc_usd",
        "deribit_price_index.eth_usd",
    ]

    while True:
        try:
            async with websockets.connect(BASE_WS, ping_interval=15, ping_timeout=10) as ws:
                token = await auth(ws)
                logger.info("Authenticated to Deribit (token len=%d)", len(token or ""))
                await subscribe(ws, channels)
                backoff = 1
                logger.info("Subscribed to trade channels: %s", channels)

                index_px = {"BTC": None, "ETH": None}

                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("method") != "subscription":
                        continue
                    channel = msg["params"]["channel"]
                    data = msg["params"]["data"]
                    if channel.startswith("deribit_price_index."):
                        if "btc" in channel:
                            index_px["BTC"] = float(data.get("price") or data.get("index_price") or data.get("estimated_delivery_price") or 0.0)
                        elif "eth" in channel:
                            index_px["ETH"] = float(data.get("price") or data.get("index_price") or data.get("estimated_delivery_price") or 0.0)
                        continue
                    if not channel.startswith("trades."):
                        continue
                    # Deribit sends trades list directly for trades.* channels
                    trades = data if isinstance(data, list) else data.get("trades") or []
                    if trades:
                        max_notional = 0.0
                        buy_count = 0
                        for trade in trades:
                            price = float(trade.get("price") or 0.0)
                            amt = float(trade.get("amount") or 0.0)
                            if trade.get("direction") == "buy":
                                buy_count += 1
                            max_notional = max(max_notional, price * amt)
                        logger.debug(
                            "Received %d trades on %s (buys %d, max notional %.2f)",
                            len(trades),
                            channel,
                            buy_count,
                            max_notional,
                        )
                    for trade in trades:
                        if trade.get("direction") != "buy":
                            continue
                        ins = trade.get("instrument_name", "")
                        # skip perps/futures; keep options only
                        if ins.endswith("-PERPETUAL") or ins.endswith("-PERP") or ".PERPETUAL" in ins:
                            continue
                        price = float(trade.get("price") or trade.get("index_price") or 0.0)
                        amt = float(trade.get("amount") or 0.0)
                        currency = ins.split("-")[0] if "-" in ins else ""
                        idx = index_px.get(currency)
                        usd_notional = (idx or 0.0) * amt
                        premium_usd = (price * amt * (idx or 0.0)) if idx else (price * amt)
                        notional = usd_notional if usd_notional else premium_usd
                        trade["_index_price"] = idx or 0.0  # for alert context
                        trade["_premium_usd"] = premium_usd
                        
                        # Check both thresholds (OR logic: either condition triggers alert)
                        notional_trigger = notional >= threshold
                        premium_trigger = premium_usd >= config.LARGE_TRADE_PREMIUM
                        
                        if notional_trigger or premium_trigger:
                            # Log which condition(s) triggered
                            triggers = []
                            if notional_trigger:
                                triggers.append(f"notional ${notional:,.0f}>=${threshold:,.0f}")
                            if premium_trigger:
                                triggers.append(f"premium ${premium_usd:,.0f}>=${config.LARGE_TRADE_PREMIUM:,.0f}")
                            
                            text = format_trade_alert(trade, notional)
                            logger.warning(text.replace("\n", " | ") + f" | Triggered by: {' & '.join(triggers)}")
                            await send_telegram(text)

        except Exception as exc:
            logger.exception("Monitor error: %s; reconnecting in %ss", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        logging.getLogger("large_trade_monitor").info("Stopped by user.")
