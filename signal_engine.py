#!/usr/bin/env python3
"""
ZenPro2026 Signal Engine — Multi-Timeframe Edition
===================================================
Timeframes : 4H · Daily · Weekly · Monthly
Signals    : MACD Zero Cross · MACD Line Cross · EMA10/Price Cross · EMA10/21 Cross
Data       : Yahoo Finance (free, no API key needed)
Delivery   : Telegram group alerts
"""

import os, json, time, math, pytz
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date

# ═══════════════════════════════════════════════════
# SECTION 1 — CREDENTIALS
# ═══════════════════════════════════════════════════
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID",  "").strip()

# ═══════════════════════════════════════════════════
# SECTION 2 — SIGNAL SWITCHES
# ═══════════════════════════════════════════════════
SIG_MACD_ZERO    = True   # MACD line crosses ZERO line
SIG_MACD_CROSS   = True   # MACD line crosses SIGNAL line
SIG_EMA10_PRICE  = True   # Price crosses EMA 10
SIG_EMA1021      = True   # EMA 10 crosses EMA 21
SIG_RSI_60       = True   # RSI crosses 60 up (buy) / 50 down (sell)
SIG_RSI_SMA      = True   # RSI crosses its own 14-period SMA

SIG_VOLUME_SPIKE = False  # Volume filter OFF — alerts fire on signal alone
                           # Set True if you only want high-volume signals

CHECK_4H    = True
CHECK_DAY   = True
CHECK_WEEK  = True
CHECK_MON   = True

SHOW_ALGO_BOX     = True
COOLDOWN_MINUTES  = 120   # 2 hours cooldown per symbol per timeframe
COOLDOWN_FILE     = "cooldown_state.json"

# ═══════════════════════════════════════════════════
# SECTION 3 — WATCHLIST
# ═══════════════════════════════════════════════════
WATCHLIST = {
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "INFY":       "INFY.NS",
    "WIPRO":      "WIPRO.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "SBIN":       "SBIN.NS",
    "NIFTY50":    "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
}

# ═══════════════════════════════════════════════════
# SECTION 4 — DATA FETCHING
# ═══════════════════════════════════════════════════
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def fetch_yahoo(symbol: str, interval: str = "1d", days: int = 600) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance. interval: 1d, 1wk, 1mo, 1h"""
    end   = int(datetime.now().timestamp())
    start = end - (days * 86400)
    url   = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
             f"?interval={interval}&period1={start}&period2={end}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        ts  = result["timestamp"]
        q   = result["indicators"]["quote"][0]
        df  = pd.DataFrame({
            "time":   pd.to_datetime(ts, unit="s"),
            "open":   q.get("open",   []),
            "high":   q.get("high",   []),
            "low":    q.get("low",    []),
            "close":  q.get("close",  []),
            "volume": q.get("volume", []),
        })
        df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [Fetch error] {symbol} {interval}: {e}")
        return pd.DataFrame()

def fetch_4h(symbol: str) -> pd.DataFrame:
    """Build 4H candles from 1H data"""
    df = fetch_yahoo(symbol, interval="1h", days=120)
    if df.empty:
        return df
    df = df.set_index("time")
    df4 = df.resample("4h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return df4.reset_index()

def fetch_daily(symbol: str)  -> pd.DataFrame:
    return fetch_yahoo(symbol, interval="1d", days=600)

def fetch_weekly(symbol: str) -> pd.DataFrame:
    return fetch_yahoo(symbol, interval="1wk", days=1500)

def fetch_monthly(symbol: str)-> pd.DataFrame:
    return fetch_yahoo(symbol, interval="1mo", days=3600)

# ═══════════════════════════════════════════════════
# SECTION 5 — INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=period-1, adjust=False).mean()
    al    = loss.ewm(com=period-1, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f  = calc_ema(close, fast)
    ema_s  = calc_ema(close, slow)
    ml     = ema_f - ema_s
    sl     = calc_ema(ml, signal)
    return ml, sl, ml - sl

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema10"]     = calc_ema(c, 10)
    df["ema21"]     = calc_ema(c, 21)
    df["ema50"]     = calc_ema(c, 50)
    df["ema200"]    = calc_ema(c, 200)
    df["rsi"]       = calc_rsi(c, 14)
    df["rsi_sma"]   = df["rsi"].rolling(14).mean()
    df["macd_line"], df["macd_sig"], df["macd_hist"] = calc_macd(c)
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = (df["volume"] / df["vol_avg20"]).round(2)
    return df

# ═══════════════════════════════════════════════════
# SECTION 6 — SIGNAL DETECTION
# ═══════════════════════════════════════════════════
def crossover(a: pd.Series, b) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a.shift(1) < b.shift(1)) & (a >= b)

def crossunder(a: pd.Series, b) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a.shift(1) > b.shift(1)) & (a <= b)

def detect_signals(df: pd.DataFrame) -> list:
    """Returns list of signals on the LAST bar only"""
    if df.empty or len(df) < 35:
        return []

    df = add_indicators(df)
    last = df.iloc[-1]
    fired = []

    # ── MACD ZERO CROSS
    if SIG_MACD_ZERO:
        if crossover(df["macd_line"], 0).iloc[-1]:
            fired.append({"type": "MACD Zero Cross",  "direction": "BUY",
                          "detail": f"MACD crossed above zero ({last['macd_line']:.3f})"})
        if crossunder(df["macd_line"], 0).iloc[-1]:
            fired.append({"type": "MACD Zero Cross",  "direction": "SELL",
                          "detail": f"MACD crossed below zero ({last['macd_line']:.3f})"})

    # ── MACD LINE CROSS
    if SIG_MACD_CROSS:
        if crossover(df["macd_line"], df["macd_sig"]).iloc[-1]:
            fired.append({"type": "MACD Line Cross",  "direction": "BUY",
                          "detail": f"MACD crossed above signal ({last['macd_line']:.3f} > {last['macd_sig']:.3f})"})
        if crossunder(df["macd_line"], df["macd_sig"]).iloc[-1]:
            fired.append({"type": "MACD Line Cross",  "direction": "SELL",
                          "detail": f"MACD crossed below signal ({last['macd_line']:.3f} < {last['macd_sig']:.3f})"})

    # ── EMA 10 / PRICE CROSS
    if SIG_EMA10_PRICE:
        if crossover(df["close"], df["ema10"]).iloc[-1]:
            fired.append({"type": "Price × EMA10",    "direction": "BUY",
                          "detail": f"Price crossed above EMA10 ({last['close']:.2f} > {last['ema10']:.2f})"})
        if crossunder(df["close"], df["ema10"]).iloc[-1]:
            fired.append({"type": "Price × EMA10",    "direction": "SELL",
                          "detail": f"Price crossed below EMA10 ({last['close']:.2f} < {last['ema10']:.2f})"})

    # ── EMA 10 / EMA 21 CROSS
    if SIG_EMA1021:
        if crossover(df["ema10"], df["ema21"]).iloc[-1]:
            fired.append({"type": "EMA10 × EMA21",    "direction": "BUY",
                          "detail": f"EMA10 crossed above EMA21 ({last['ema10']:.2f} > {last['ema21']:.2f})"})
        if crossunder(df["ema10"], df["ema21"]).iloc[-1]:
            fired.append({"type": "EMA10 × EMA21",    "direction": "SELL",
                          "detail": f"EMA10 crossed below EMA21 ({last['ema10']:.2f} < {last['ema21']:.2f})"})

    # ── RSI 60/50 MOMENTUM
    if SIG_RSI_60:
        if crossover(df["rsi"], 60).iloc[-1]:
            fired.append({"type": "RSI Momentum",     "direction": "BUY",
                          "detail": f"RSI crossed above 60 ({last['rsi']:.1f})"})
        if crossunder(df["rsi"], 50).iloc[-1]:
            fired.append({"type": "RSI Momentum",     "direction": "SELL",
                          "detail": f"RSI crossed below 50 ({last['rsi']:.1f})"})

    # ── RSI × SMA CROSS (early signal)
    if SIG_RSI_SMA:
        if crossover(df["rsi"], df["rsi_sma"]).iloc[-1]:
            fired.append({"type": "RSI × SMA",        "direction": "BUY",
                          "detail": f"RSI crossed above its SMA ({last['rsi']:.1f} > {last['rsi_sma']:.1f})"})
        if crossunder(df["rsi"], df["rsi_sma"]).iloc[-1]:
            fired.append({"type": "RSI × SMA",        "direction": "SELL",
                          "detail": f"RSI crossed below its SMA ({last['rsi']:.1f} < {last['rsi_sma']:.1f})"})

    # ── VOLUME FILTER (optional)
    if SIG_VOLUME_SPIKE and fired:
        if last["vol_ratio"] < 2.0:
            fired = []  # suppress — no volume confirmation

    return fired, last

# ═══════════════════════════════════════════════════
# SECTION 7 — ALGO BOX SCORE
# ═══════════════════════════════════════════════════
def tf_score(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 30:
        return {"l1":0,"l2":0,"l3":0,"l4":0,"score":0}
    df = add_indicators(df)
    last = df.iloc[-1]
    l1 = 1 if last["rsi"] > last["rsi_sma"] else 0
    l2 = 1 if last["macd_line"] > last["macd_sig"] else 0
    l3 = 1 if last["rsi"] > 50 else 0
    l4 = 1 if last["ema10"] > last["ema21"] else 0
    return {"l1":l1,"l2":l2,"l3":l3,"l4":l4,"score":l1+l2+l3+l4}

def progress_bar(sc: dict) -> str:
    return "".join("█" if sc.get(k) else "░" for k in ["l1","l2","l3","l4"])

def algo_box(daily_df: pd.DataFrame) -> dict:
    if daily_df.empty or len(daily_df) < 200:
        return {"state":"NO DATA","total":0}
    # Resample from daily
    daily_df = daily_df.set_index("time") if "time" in daily_df.columns else daily_df
    weekly  = daily_df.resample("W").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
    monthly = daily_df.resample("ME").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
    daily_df = daily_df.reset_index()

    d = tf_score(daily_df.rename(columns={"index":"time"}) if "time" not in daily_df.columns else daily_df)
    w = tf_score(weekly)
    m = tf_score(monthly)
    total = m["score"]*3 + w["score"]*2 + d["score"]*1

    state = ("STRONG BUY" if total>=20 else "BUY" if total>=15
             else "BUY DIP" if total>=10 else "HOLD" if total>=6 else "CAUTION")
    return {"state":state,"total":total,"d":d,"w":w,"m":m}

# ═══════════════════════════════════════════════════
# SECTION 8 — TELEGRAM
# ═══════════════════════════════════════════════════
def send_telegram(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"  [Telegram error] {e}")
        return False

def format_alert(symbol: str, tf_label: str, signals: list,
                 last_row, ab: dict) -> str:
    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST).strftime("%d %b %Y  %H:%M IST")

    directions = {s["direction"] for s in signals}
    if "BUY" in directions and "SELL" not in directions:
        emoji, dir_text = "🟢", "BUY"
    elif "SELL" in directions and "BUY" not in directions:
        emoji, dir_text = "🔴", "SELL / EXIT"
    else:
        emoji, dir_text = "🟡", "MIXED"

    state_emoji = {
        "STRONG BUY":"⬆","BUY":"↑","BUY DIP":"↗",
        "HOLD":"◆","CAUTION":"⚠","NO DATA":"—"
    }.get(ab.get("state",""),  "—")

    sig_lines = "\n".join(
        f"  {'▲' if s['direction']=='BUY' else '▼'} <b>{s['type']}</b> — {s['detail']}"
        for s in signals
    )

    # Progress bars
    d = ab.get("d", {}); w = ab.get("w", {}); m = ab.get("m", {})

    price  = float(last_row.get("close", 0))
    ema10  = float(last_row.get("ema10", 0))
    ema21  = float(last_row.get("ema21", 0))
    rsi_v  = float(last_row.get("rsi", 0))
    macd_v = float(last_row.get("macd_line", 0))
    vol_r  = float(last_row.get("vol_ratio", 1))

    rsi_zone  = "Overbought ⚠" if rsi_v>=70 else "Above 50 ✓" if rsi_v>=50 else "Below 50 ✗"
    macd_zone = "Above zero ✓" if macd_v>0 else "Below zero ✗"

    return (
        f"{emoji} <b>ZenPro2026 — {symbol}</b>  [{tf_label}]  {dir_text}\n"
        f"{'─'*34}\n"
        f"<b>Signals fired:</b>\n{sig_lines}\n"
        f"{'─'*34}\n"
        f"{state_emoji} <b>Algo Box: {ab.get('state','—')}</b>  [{ab.get('total',0)}/24]\n"
        f"M:{progress_bar(m)}  W:{progress_bar(w)}  D:{progress_bar(d)}\n"
        f"M×3={m.get('score',0)*3}  W×2={w.get('score',0)*2}  D×1={d.get('score',0)}\n"
        f"{'─'*34}\n"
        f"Price : ₹{price:.2f}\n"
        f"EMA10 : {ema10:.2f}  |  EMA21: {ema21:.2f}\n"
        f"MACD  : {macd_v:+.3f} ({macd_zone})\n"
        f"RSI   : {rsi_v:.1f} ({rsi_zone})\n"
        f"Vol   : {vol_r:.1f}× avg\n"
        f"{'─'*34}\n"
        f"<i>{now}</i>\n"
        f"<i>zeninvestor.in — ZenPro2026</i>"
    )

# ═══════════════════════════════════════════════════
# SECTION 9 — COOLDOWN
# ═══════════════════════════════════════════════════
def load_cooldown() -> dict:
    try:
        if os.path.exists(COOLDOWN_FILE):
            with open(COOLDOWN_FILE) as f:
                return json.load(f)
    except:
        pass
    return {}

def save_cooldown(state: dict):
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"  [Cooldown save error] {e}")

def in_cooldown(key: str, state: dict) -> bool:
    ts = state.get(key)
    if not ts:
        return False
    return (time.time() - ts) / 60 < COOLDOWN_MINUTES

def update_cooldown(key: str, state: dict):
    state[key] = time.time()

# ═══════════════════════════════════════════════════
# SECTION 10 — MARKET HOURS
# ═══════════════════════════════════════════════════
def is_market_open() -> bool:
    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    c = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return o <= now <= c

# ═══════════════════════════════════════════════════
# SECTION 11 — MAIN ENGINE
# ═══════════════════════════════════════════════════
def run():
    print("=" * 52)
    print(f"ZenPro2026 Signal Engine  —  {datetime.now()}")
    print("=" * 52)

    if not is_market_open():
        print("Market closed — skipping run.")
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: Telegram credentials not set.")
        return

    cooldown = load_cooldown()
    alerts_sent = 0

    # Define timeframes to check
    TF_MAP = []
    if CHECK_4H:   TF_MAP.append(("4H",  fetch_4h))
    if CHECK_DAY:  TF_MAP.append(("1D",  fetch_daily))
    if CHECK_WEEK: TF_MAP.append(("1W",  fetch_weekly))
    if CHECK_MON:  TF_MAP.append(("1M",  fetch_monthly))

    for symbol, yahoo in WATCHLIST.items():
        print(f"\nProcessing {symbol}...")

        # Fetch daily once for Algo Box
        daily_df = fetch_daily(yahoo)
        ab = algo_box(daily_df) if SHOW_ALGO_BOX and not daily_df.empty else {"state":"—","total":0,"d":{},"w":{},"m":{}}

        for tf_label, fetch_fn in TF_MAP:
            cd_key = f"{symbol}_{tf_label}"

            if in_cooldown(cd_key, cooldown):
                print(f"  [{tf_label}] Cooldown — skipping")
                continue

            df = fetch_fn(yahoo)
            if df.empty or len(df) < 35:
                print(f"  [{tf_label}] Not enough data — skipping")
                continue

            result = detect_signals(df)
            if result is None:
                continue
            signals, last_row = result

            if not signals:
                print(f"  [{tf_label}] No signals")
                continue

            # Signals fired!
            print(f"  [{tf_label}] 🔔 {[s['type']+' '+s['direction'] for s in signals]}")
            msg = format_alert(symbol, tf_label, signals, last_row, ab)

            if send_telegram(msg):
                print(f"  [{tf_label}] ✅ Alert sent!")
                update_cooldown(cd_key, cooldown)
                alerts_sent += 1
            else:
                print(f"  [{tf_label}] ❌ Telegram failed")

            time.sleep(1)

        time.sleep(2)

    save_cooldown(cooldown)
    print(f"\n{'='*52}")
    print(f"Done. {alerts_sent} alert(s) sent.")

if __name__ == "__main__":
    run()
