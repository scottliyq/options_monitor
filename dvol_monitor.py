#!/usr/bin/env python3
"""
Deribit DVOL (Volatility Index) Monitor for BTC and ETH options.

Monitors:
  1. IV Pulse: 1h DVOL rise >= 5%
  2. Price Filter: 1h price drop <= -2.5%
  
When both conditions are met simultaneously, sends highlighted Telegram alert.

Usage:
  python dvol_monitor.py

Environment:
  TELEGRAM_BOT_TOKEN  Telegram bot token for alerts
  TELEGRAM_CHAT_ID    Telegram chat ID for alerts
  DERIBIT_TESTNET     Set to "1" to use Deribit testnet
  LOG_LEVEL           Logging level (default: INFO)
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import aiohttp
from dotenv import load_dotenv

import config

load_dotenv()

# Deribit API endpoints
BASE_URL = (
    "https://test.deribit.com/api/v2"
    if os.getenv("DERIBIT_TESTNET")
    else "https://www.deribit.com/api/v2"
)


class RollingWindow:
    """Fixed-duration rolling window of (timestamp, value) points."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.data: Deque[Tuple[float, float]] = deque()

    def add(self, ts: float, value: float) -> None:
        """Add a data point and remove expired ones."""
        self.data.append((ts, value))
        cutoff = ts - self.seconds
        while self.data and self.data[0][0] < cutoff:
            self.data.popleft()

    def pct_change(self) -> float:
        """Calculate percentage change from first to last point in window."""
        if len(self.data) < 2:
            return 0.0
        start_val = self.data[0][1]
        end_val = self.data[-1][1]
        if start_val == 0:
            return 0.0
        return (end_val - start_val) / start_val

    def get_latest(self) -> Optional[float]:
        """Get the latest value in the window."""
        if not self.data:
            return None
        return self.data[-1][1]

    def get_oldest(self) -> Optional[float]:
        """Get the oldest value in the window."""
        if not self.data:
            return None
        return self.data[0][1]


async def send_telegram(text: str, parse_mode: str = "HTML") -> None:
    """Send message via Telegram Bot API."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.getLogger("dvol_monitor").warning(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set, skipping Telegram notification"
        )
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, timeout=5) as resp:
                if resp.status != 200:
                    logging.getLogger("dvol_monitor").error(
                        f"Telegram API error: status={resp.status}, text={await resp.text()}"
                    )
        except Exception as exc:
            logging.getLogger("dvol_monitor").exception(f"Failed to send Telegram message: {exc}")


def format_dvol_alert(
    currency: str,
    dvol_change: float,
    price_change: float,
    current_dvol: float,
    current_price: float,
    old_dvol: float,
    old_price: float,
) -> str:
    """Format DVOL alert message with HTML highlighting."""
    # Emoji based on currency
    emoji = "₿" if currency == "BTC" else "Ξ"
    
    # Color highlighting for key values
    dvol_rise_pct = dvol_change * 100
    price_drop_pct = price_change * 100
    
    message = (
        f"🚨 <b><u>{emoji} {currency} DVOL IV PULSE ALERT</u></b> 🚨\n\n"
        f"<b>📊 DVOL (Volatility Index):</b>\n"
        f"  • Current: <code>{current_dvol:.2f}</code>\n"
        f"  • 1h ago: <code>{old_dvol:.2f}</code>\n"
        f"  • <b>1h Change: <u>+{dvol_rise_pct:.2f}%</u></b> ⬆️\n\n"
        f"<b>💰 {currency} Price:</b>\n"
        f"  • Current: <code>${current_price:,.2f}</code>\n"
        f"  • 1h ago: <code>${old_price:,.2f}</code>\n"
        f"  • <b>1h Change: <u>{price_drop_pct:.2f}%</u></b> ⬇️\n\n"
        f"<b>⚠️ Alert Conditions:</b>\n"
        f"  ✅ IV Pulse: DVOL rise ≥ {config.DVOL_RISE_THRESHOLD * 100:.1f}%\n"
        f"  ✅ Price Drop: Price drop ≤ {config.PRICE_DROP_THRESHOLD * 100:.1f}%\n\n"
        f"<i>Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
    )
    
    return message


def format_dvol_snapshot(
    btc_dvol: float,
    btc_price: float,
    btc_dvol_change: float,
    btc_price_change: float,
    eth_dvol: float,
    eth_price: float,
    eth_dvol_change: float,
    eth_price_change: float,
) -> str:
    """Format DVOL snapshot message (regular status update, not alert)."""
    # Format changes with +/- sign
    btc_dvol_str = f"{btc_dvol_change * 100:+.2f}%" if btc_dvol_change != 0 else "N/A"
    btc_price_str = f"{btc_price_change * 100:+.2f}%" if btc_price_change != 0 else "N/A"
    eth_dvol_str = f"{eth_dvol_change * 100:+.2f}%" if eth_dvol_change != 0 else "N/A"
    eth_price_str = f"{eth_price_change * 100:+.2f}%" if eth_price_change != 0 else "N/A"
    
    message = (
        f"📸 <b>DVOL Monitor Snapshot</b>\n\n"
        f"<b>₿ BTC:</b>\n"
        f"  DVOL: <code>{btc_dvol:.2f}</code> ({btc_dvol_str} 1h)\n"
        f"  Price: <code>${btc_price:,.2f}</code> ({btc_price_str} 1h)\n\n"
        f"<b>Ξ ETH:</b>\n"
        f"  DVOL: <code>{eth_dvol:.2f}</code> ({eth_dvol_str} 1h)\n"
        f"  Price: <code>${eth_price:,.2f}</code> ({eth_price_str} 1h)\n\n"
        f"<i>Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
    )
    
    return message


async def fetch_dvol(session: aiohttp.ClientSession, currency: str) -> Optional[float]:
    """Fetch DVOL (Volatility Index) for a currency using /public/get_index_price."""
    try:
        # Correct DVOL index names for Deribit: btcdvol_usdc, ethdvol_usdc
        index_name = f"{currency.lower()}dvol_usdc"
        url = f"{BASE_URL}/public/get_index_price"
        params = {"index_name": index_name}
        
        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status != 200:
                text = await resp.text()
                logging.getLogger("dvol_monitor").debug(
                    f"Failed to fetch DVOL for {currency}: status={resp.status}, body={text[:200]}"
                )
                return None
            data = await resp.json()
            
            # Log full response
            logging.getLogger("dvol_monitor").debug(
                f"DVOL API response for {currency} ({index_name}): {json.dumps(data)}"
            )
            
            # Extract DVOL from result
            result = data.get("result", {})
            
            # The result should contain 'index_price' field
            dvol_value = result.get("index_price") or result.get("edp")
            
            if dvol_value is not None:
                logging.getLogger("dvol_monitor").info(
                    f"Successfully fetched {currency} DVOL: {float(dvol_value):.2f}"
                )
                return float(dvol_value)
            
            logging.getLogger("dvol_monitor").warning(
                f"DVOL value not found in response for {currency}: {result}"
            )
            return None
    except Exception as exc:
        logging.getLogger("dvol_monitor").error(f"Exception fetching DVOL for {currency}: {exc}")
        return None


async def fetch_price_index(session: aiohttp.ClientSession, currency: str) -> Optional[float]:
    """Fetch price index for a currency using /public/get_index_price."""
    try:
        url = f"{BASE_URL}/public/get_index_price"
        params = {"index_name": f"{currency.lower()}_usd"}
        
        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status != 200:
                text = await resp.text()
                logging.getLogger("dvol_monitor").debug(
                    f"Failed to fetch price for {currency}: status={resp.status}, body={text[:200]}"
                )
                return None
            data = await resp.json()
            
            # Log full response
            logging.getLogger("dvol_monitor").debug(
                f"Price API response for {currency}: {json.dumps(data)}"
            )
            
            # Extract price from result
            result = data.get("result", {})
            price_value = result.get("index_price")
            
            if price_value is not None:
                logging.getLogger("dvol_monitor").info(
                    f"Successfully fetched {currency} Price: ${float(price_value):,.2f}"
                )
                return float(price_value)
            
            logging.getLogger("dvol_monitor").warning(
                f"Price value not found in response for {currency}: {result}"
            )
            return None
    except Exception as exc:
        logging.getLogger("dvol_monitor").error(f"Exception fetching price for {currency}: {exc}")
        return None


async def monitor() -> None:
    """Main monitoring loop using HTTP polling."""
    logger = logging.getLogger("dvol_monitor")
    
    # Rolling windows for DVOL and price (1 hour window)
    dvol_windows: Dict[str, RollingWindow] = {
        "BTC": RollingWindow(config.DVOL_WINDOW_SEC),
        "ETH": RollingWindow(config.DVOL_WINDOW_SEC),
    }
    price_windows: Dict[str, RollingWindow] = {
        "BTC": RollingWindow(config.DVOL_WINDOW_SEC),
        "ETH": RollingWindow(config.DVOL_WINDOW_SEC),
    }
    
    # Track last alert time to avoid spam (cooldown: 10 minutes)
    last_alert_time: Dict[str, float] = {"BTC": 0.0, "ETH": 0.0}
    alert_cooldown = 600  # 10 minutes in seconds
    
    # Track last snapshot time for periodic reporting
    last_snapshot_time = 0.0
    snapshot_interval = config.DVOL_SNAPSHOT_INTERVAL_SEC
    
    # Polling interval (10 seconds)
    poll_interval = 10
    
    logger.info("Starting DVOL monitor (HTTP polling mode)...")
    logger.info(f"Thresholds: DVOL rise >= {config.DVOL_RISE_THRESHOLD * 100:.1f}%, "
                f"Price drop <= {config.PRICE_DROP_THRESHOLD * 100:.1f}%")
    logger.info(f"Window: {config.DVOL_WINDOW_SEC}s ({config.DVOL_WINDOW_SEC / 3600:.1f}h)")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info(f"Snapshot interval: {snapshot_interval}s ({snapshot_interval / 3600:.1f}h)")
    
    poll_count = 0
    initial_snapshot_sent = False
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                ts = time.time()
                poll_count += 1
                
                # Fetch data for both currencies
                for currency in ["BTC", "ETH"]:
                    # Fetch DVOL
                    dvol_value = await fetch_dvol(session, currency)
                    if dvol_value is not None and dvol_value > 0:
                        dvol_windows[currency].add(ts, dvol_value)
                        if poll_count % 6 == 0:  # Log every minute (6 * 10s)
                            logger.debug(f"{currency} DVOL: {dvol_value:.2f}, "
                                       f"points: {len(dvol_windows[currency].data)}")
                    
                    # Fetch price index
                    price_value = await fetch_price_index(session, currency)
                    if price_value is not None and price_value > 0:
                        price_windows[currency].add(ts, price_value)
                        if poll_count % 6 == 0:
                            logger.debug(f"{currency} Price: ${price_value:,.2f}, "
                                       f"points: {len(price_windows[currency].data)}")
                
                # Send initial snapshot on first successful data fetch
                if not initial_snapshot_sent:
                    btc_has_data = len(dvol_windows["BTC"].data) > 0 and len(price_windows["BTC"].data) > 0
                    eth_has_data = len(dvol_windows["ETH"].data) > 0 and len(price_windows["ETH"].data) > 0
                    
                    if btc_has_data and eth_has_data:
                        # Send initial snapshot
                        snapshot_text = format_dvol_snapshot(
                            btc_dvol=dvol_windows["BTC"].get_latest() or 0.0,
                            btc_price=price_windows["BTC"].get_latest() or 0.0,
                            btc_dvol_change=dvol_windows["BTC"].pct_change() if len(dvol_windows["BTC"].data) >= 2 else 0.0,
                            btc_price_change=price_windows["BTC"].pct_change() if len(price_windows["BTC"].data) >= 2 else 0.0,
                            eth_dvol=dvol_windows["ETH"].get_latest() or 0.0,
                            eth_price=price_windows["ETH"].get_latest() or 0.0,
                            eth_dvol_change=dvol_windows["ETH"].pct_change() if len(dvol_windows["ETH"].data) >= 2 else 0.0,
                            eth_price_change=price_windows["ETH"].pct_change() if len(price_windows["ETH"].data) >= 2 else 0.0,
                        )
                        logger.info("Sending initial snapshot...")
                        await send_telegram(snapshot_text)
                        initial_snapshot_sent = True
                        last_snapshot_time = ts
                
                # Send periodic snapshot every 4 hours
                if ts - last_snapshot_time >= snapshot_interval and initial_snapshot_sent:
                    snapshot_text = format_dvol_snapshot(
                        btc_dvol=dvol_windows["BTC"].get_latest() or 0.0,
                        btc_price=price_windows["BTC"].get_latest() or 0.0,
                        btc_dvol_change=dvol_windows["BTC"].pct_change() if len(dvol_windows["BTC"].data) >= 2 else 0.0,
                        btc_price_change=price_windows["BTC"].pct_change() if len(price_windows["BTC"].data) >= 2 else 0.0,
                        eth_dvol=dvol_windows["ETH"].get_latest() or 0.0,
                        eth_price=price_windows["ETH"].get_latest() or 0.0,
                        eth_dvol_change=dvol_windows["ETH"].pct_change() if len(dvol_windows["ETH"].data) >= 2 else 0.0,
                        eth_price_change=price_windows["ETH"].pct_change() if len(price_windows["ETH"].data) >= 2 else 0.0,
                    )
                    logger.info("Sending periodic snapshot...")
                    await send_telegram(snapshot_text)
                    last_snapshot_time = ts
                
                # Log status every 5 minutes (30 polls * 10s)
                if poll_count % 30 == 0:
                    for currency in ["BTC", "ETH"]:
                        dvol_window = dvol_windows[currency]
                        price_window = price_windows[currency]
                        if len(dvol_window.data) >= 2 and len(price_window.data) >= 2:
                            dvol_change = dvol_window.pct_change()
                            price_change = price_window.pct_change()
                            current_dvol = dvol_window.get_latest() or 0.0
                            current_price = price_window.get_latest() or 0.0
                            
                            logger.info(
                                f"{currency}: DVOL={current_dvol:.2f} ({dvol_change * 100:+.2f}%), "
                                f"Price=${current_price:,.2f} ({price_change * 100:+.2f}%)"
                            )
                
                # Check alert conditions for each currency
                for currency in ["BTC", "ETH"]:
                    dvol_window = dvol_windows[currency]
                    price_window = price_windows[currency]
                    
                    # Need at least 2 data points (start and end of window)
                    if len(dvol_window.data) < 2 or len(price_window.data) < 2:
                        continue
                    
                    # Calculate changes
                    dvol_change = dvol_window.pct_change()
                    price_change = price_window.pct_change()
                    
                    # Check if both conditions are met
                    dvol_rise_triggered = dvol_change >= config.DVOL_RISE_THRESHOLD
                    price_drop_triggered = price_change <= config.PRICE_DROP_THRESHOLD
                    
                    if dvol_rise_triggered and price_drop_triggered:
                        # Check cooldown to avoid spam
                        time_since_last_alert = ts - last_alert_time[currency]
                        if time_since_last_alert < alert_cooldown:
                            logger.debug(
                                f"{currency} alert triggered but in cooldown "
                                f"({time_since_last_alert:.0f}s / {alert_cooldown}s)"
                            )
                            continue
                        
                        # Get current and old values
                        current_dvol = dvol_window.get_latest() or 0.0
                        old_dvol = dvol_window.get_oldest() or 0.0
                        current_price = price_window.get_latest() or 0.0
                        old_price = price_window.get_oldest() or 0.0
                        
                        # Format and send alert
                        alert_text = format_dvol_alert(
                            currency=currency,
                            dvol_change=dvol_change,
                            price_change=price_change,
                            current_dvol=current_dvol,
                            current_price=current_price,
                            old_dvol=old_dvol,
                            old_price=old_price,
                        )
                        
                        # Log alert
                        logger.warning(
                            f"🚨 {currency} DVOL ALERT: "
                            f"DVOL +{dvol_change * 100:.2f}%, "
                            f"Price {price_change * 100:.2f}%"
                        )
                        
                        # Send to Telegram
                        await send_telegram(alert_text)
                        
                        # Update last alert time
                        last_alert_time[currency] = ts
                
                # Wait for next poll
                await asyncio.sleep(poll_interval)
                
            except Exception as exc:
                logger.exception(f"Monitor error: {exc}; continuing in {poll_interval}s")
                await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        logging.getLogger("dvol_monitor").info("Stopped by user.")
