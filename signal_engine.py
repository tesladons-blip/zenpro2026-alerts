#!/usr/bin/env python3
"""
ZenPro2026 Signal Engine
========================
Uses Yahoo Finance (free, no API key needed) for market data.
Runs via GitHub Actions every 5 min during market hours (9:15-15:30 IST).
Calculates signals → sends Telegram alerts.
"""

import os, json, time, math, pytz
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import requests

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CREDENTIALS (only Telegram needed now!)
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID",  "").strip()

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — SIGNAL SWITCHES
# ═══════════════════════════════════════════════════════════════

SIG_MACD_ZERO_CROSS  = True
SIG_MACD_LINE_CROSS  = False
SIG_EMA10_21_CROSS   = True
SIG_RSI_MOMENTUM     = True
SIG_RSI_50_ENTRY     = False
SIG_RSI_EARLY        = False
SIG_ALGO_BOX         = True
SIG_VOLUME_SPIKE     = True

ALGO_STRONG_BUY_MIN  = 20
ALGO_BUY_MIN         = 15
ALGO_BUY_DIP_MIN     = 10
ALGO_HOLD_MIN        = 6

VOL_SPIKE_MULTIPLIER = 2.0
COOLDOWN_MINUTES     = 30
COOLDOWN_FILE        = "cooldown_state.json"

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — WATCHLIST
# Yahoo Finance symbols for NSE stocks — add .NS at the end
# For indices use: ^NSEI (Nifty50), ^NSEBANK (BankNifty)
# ═══════════════════════════════════════════════════════════════

WATCHLIST = {
    "RELIANCE":  "RELIANCE.NS",
    "TCS":       "TCS.NS",
    "HDFCBANK":  "HDFCBANK.NS",
    "INFY":      "INFY.NS",
    "WIPRO":     "WIPRO.NS",
    "NIFTY50":   "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — YAHOO FINANCE DATA FETCHER (FREE!)
# No API key, no subscription, no daily token refresh needed
# ═══════════════════════════════════════════════════════════════

def fetch_daily_ohlcv(symbol: str, days: int = 600) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Yahoo Finance.
    Returns DataFrame with columns: date, open, high, low, close, volume
    """
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d"
        f"&period1={int(datetime.combine(start_dt, datetime.min.time()).timestamp())}"
        f"&period2={int(datetime.combine(end_dt,   datetime.min.time()).timestamp())}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        ohlcv      = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "date":   pd.to_datetime(timestamps, unit="s"),
            "open":   ohlcv.get("open",   []),
            "high":   ohlcv.get("high",   []),
            "low":    ohlcv.get("low",    []),
            "close":  ohlcv.get("close",  []),
            "volume": ohlcv.get("volume", []),
        })

        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"  [Yahoo fetch error] {symbol}: {e}")
        return pd.DataFrame()

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.set_index("date")
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = calc_ema(close, fast)
    ema_slow    = calc_ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def crossover(a: pd.Series, b) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a.shift(1) < b.shift(1)) & (a >= b)

def crossunder(a: pd.Series, b) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a.shift(1) > b.shift(1)) & (a <= b)

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].copy()
    v = df["volume"].copy()

    df["ema10"]  = calc_ema(c, 10)
    df["ema21"]  = calc_ema(c, 21)
    df["ema50"]  = calc_ema(c, 50)
    df["ema100"] = calc_ema(c, 100)
    df["ema150"] = calc_ema(c, 150)
    df["ema200"] = calc_ema(c, 200)
    df["sma200"] = c.rolling(200).mean()
    df["ema13"]  = calc_ema(c, 13)
    df["ema27"]  = calc_ema(c, 27)

    df["rsi"]     = calc_rsi(c, 14)
    df["rsi_sma"] = df["rsi"].rolling(14).mean()

    df["rsi_rising"]  = df["rsi"] > df["rsi"].shift(1)
    df["rsi_falling"] = df["rsi"] < df["rsi"].shift(1)

    df["macd_line"], df["macd_sig"], df["macd_hist"] = calc_macd(c)

    df["vol_avg20"] = v.rolling(20).mean()
    df["vol_spike"] = v > (df["vol_avg20"] * VOL_SPIKE_MULTIPLIER)
    df["vol_ratio"] = (v / df["vol_avg20"]).round(2)

    df["high252"]  = df["high"].rolling(252).max()
    df["near_ath"] = c >= df["high252"] * 0.97

    df["cross_macdz_up"]    = crossover(df["macd_line"], 0)
    df["cross_macdz_dn"]    = crossunder(df["macd_line"], 0)
    df["cross_macdl_up"]    = crossover(df["macd_line"], df["macd_sig"])
    df["cross_macdl_dn"]    = crossunder(df["macd_line"], df["macd_sig"])
    df["cross_ema10_21_up"] = crossover(df["ema10"], df["ema21"])
    df["cross_ema10_21_dn"] = crossunder(df["ema10"], df["ema21"])
    df["cross_rsi_sma_up"]  = crossover(df["rsi"], df["rsi_sma"])
    df["cross_rsi_sma_dn"]  = crossunder(df["rsi"], df["rsi_sma"])
    df["cross_rsi_50_up"]   = crossover(df["rsi"], 50) & df["rsi_rising"]
    df["cross_rsi_50_dn"]   = crossunder(df["rsi"], 50) & df["rsi_falling"]
    df["cross_rsi_60_up"]   = crossover(df["rsi"], 60) & df["rsi_rising"]
    df["cross_rsi_50_sell"] = crossunder(df["rsi"], 50) & df["rsi_falling"]

    return df

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — ALGO SIGNAL BOX
# ═══════════════════════════════════════════════════════════════

def progress_bar(sc: dict) -> str:
    return "".join("█" if sc.get(k) else "░" for k in ["l1","l2","l3","l4"])

def calc_tf_score(df_daily: pd.DataFrame, rule: str) -> dict:
    if df_daily.empty:
        return {"l1": 0, "l2": 0, "l3": 0, "l4": 0, "score": 0}

    df_tf = resample_ohlcv(
        df_daily[["date","open","high","low","close","volume"]], rule
    )
    if len(df_tf) < 30:
        return {"l1": 0, "l2": 0, "l3": 0, "l4": 0, "score": 0}

    df_tf = calc_indicators(df_tf)
    last  = df_tf.iloc[-1]

    l1 = 1 if last["rsi"] > last["rsi_sma"] else 0
    l2 = 1 if last["macd_line"] > last["macd_sig"] else 0
    l3 = 1 if last["rsi"] > 50 else 0
    l4 = 1 if last["ema10"] > last["ema21"] else 0

    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4,
            "score": l1+l2+l3+l4}

def algo_signal_box(df_daily: pd.DataFrame) -> dict:
    if df_daily.empty or len(df_daily) < 250:
        return {"sig_state": "NO DATA", "total_score": 0}

    df_i   = calc_indicators(df_daily.copy())
    last_d = df_i.iloc[-1]

    d_sc = {
        "l1": 1 if last_d["rsi"] > last_d["rsi_sma"] else 0,
        "l2": 1 if last_d["macd_line"] > last_d["macd_sig"] else 0,
        "l3": 1 if last_d["rsi"] > 50 else 0,
        "l4": 1 if last_d["ema10"] > last_d["ema21"] else 0,
    }
    d_sc["score"] = sum(d_sc[k] for k in ["l1","l2","l3","l4"])

    w_sc = calc_tf_score(df_daily, "W")
    m_sc = calc_tf_score(df_daily, "ME")

    total_score = (m_sc["score"] * 3) + (w_sc["score"] * 2) + (d_sc["score"] * 1)

    near_ath = bool(last_d["near_ath"])
    eff_score = max(total_score, ALGO_STRONG_BUY_MIN) \
        if near_ath and total_score >= ALGO_BUY_MIN else total_score

    above_d200 = last_d["close"] > last_d["sma200"] \
        if not math.isnan(last_d["sma200"]) else True

    df_w = resample_ohlcv(
        df_daily[["date","open","high","low","close","volume"]], "W"
    )
    df_w["sma200"] = df_w["close"].rolling(200).mean()
    above_w200 = float(df_w["close"].iloc[-1]) > float(df_w["sma200"].iloc[-1]) \
        if len(df_w) >= 200 else True

    trend_down = not above_d200 and not above_w200
    trend_weak = not above_d200 and above_w200

    h1_rsi_ob     = last_d["rsi"] >= 70
    h1_macd_exh   = (last_d["macd_line"] > 0) and \
                    (last_d["macd_line"] < last_d["macd_sig"])
    h1_exhaustion = h1_rsi_ob or h1_macd_exh

    if trend_down:
        sig_state = "TREND DOWN"
    elif trend_weak:
        sig_state = "TREND WEAK"
    elif eff_score >= ALGO_STRONG_BUY_MIN:
        sig_state = "STRONG BUY"
    elif eff_score >= ALGO_BUY_MIN:
        sig_state = "BUY"
    elif eff_score >= ALGO_BUY_DIP_MIN:
        sig_state = "BUY DIP"
    elif eff_score >= ALGO_HOLD_MIN:
        sig_state = "HOLD"
    else:
        sig_state = "CAUTION"

    return {
        "sig_state":     sig_state,
        "total_score":   total_score,
        "eff_score":     eff_score,
        "m_sc":          m_sc,
        "w_sc":          w_sc,
        "d_sc":          d_sc,
        "near_ath":      near_ath,
        "h1_rsi_ob":     h1_rsi_ob,
        "h1_macd_exh":   h1_macd_exh,
        "h1_exhaustion": h1_exhaustion,
        "trend_down":    trend_down,
        "trend_weak":    trend_weak,
    }

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_signals(df: pd.DataFrame):
    if df.empty or len(df) < 30:
        return [], None

    df   = calc_indicators(df)
    last = df.iloc[-1]
    fired = []

    if SIG_MACD_ZERO_CROSS:
        if last["cross_macdz_up"]:
            fired.append({"type": "MACD Zero Cross UP",  "direction": "BUY"})
        if last["cross_macdz_dn"]:
            fired.append({"type": "MACD Zero Cross DN",  "direction": "SELL"})

    if SIG_MACD_LINE_CROSS:
        if last["cross_macdl_up"]:
            fired.append({"type": "MACD Line Cross UP",  "direction": "BUY"})
        if last["cross_macdl_dn"]:
            fired.append({"type": "MACD Line Cross DN",  "direction": "SELL"})

    if SIG_EMA10_21_CROSS:
        if last["cross_ema10_21_up"]:
            fired.append({"type": "EMA10 x EMA21 UP",   "direction": "BUY"})
        if last["cross_ema10_21_dn"]:
            fired.append({"type": "EMA10 x EMA21 DN",   "direction": "SELL"})

    if SIG_RSI_MOMENTUM:
        if last["cross_rsi_60_up"]:
            fired.append({"type": "RSI Momentum BUY (crossed 60)", "direction": "BUY"})
        if last["cross_rsi_50_sell"]:
            fired.append({"type": "RSI Momentum SELL (crossed 50)","direction": "SELL"})

    if SIG_RSI_50_ENTRY:
        if last["cross_rsi_50_up"]:
            fired.append({"type": "RSI 50 Entry BUY",   "direction": "BUY"})
        if last["cross_rsi_50_dn"]:
            fired.append({"type": "RSI 50 Entry SELL",  "direction": "SELL"})

    if SIG_RSI_EARLY:
        if last["cross_rsi_sma_up"]:
            fired.append({"type": "RSI Early Entry BUY",  "direction": "BUY"})
        if last["cross_rsi_sma_dn"]:
            fired.append({"type": "RSI Early Exit SELL",  "direction": "SELL"})

    if SIG_VOLUME_SPIKE and fired:
        if not last["vol_spike"]:
            fired = []

    return fired, last

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — TELEGRAM MESSAGING
# ═══════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"  [Telegram error] {e}")
        return False

def format_alert(symbol: str, signals: list, last_row,
                 algo: dict, vol_ratio: float) -> str:
    IST     = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST).strftime("%d %b %Y  %H:%M IST")

    directions = set(s["direction"] for s in signals)
    if "BUY" in directions and "SELL" not in directions:
        dir_emoji, dir_text = "🟢", "BUY"
    elif "SELL" in directions and "BUY" not in directions:
        dir_emoji, dir_text = "🔴", "SELL / EXIT"
    else:
        dir_emoji, dir_text = "🟡", "MIXED"

    state_emoji = {
        "STRONG BUY": "⬆", "BUY": "↑", "BUY DIP": "↗",
        "HOLD": "◆", "CAUTION": "⚠", "TREND WEAK": "⚠",
        "TREND DOWN": "↓", "NO DATA": "—",
    }.get(algo.get("sig_state", ""), "—")

    sig_state   = algo.get("sig_state", "—")
    total_score = algo.get("total_score", 0)
    m_sc = algo.get("m_sc", {})
    w_sc = algo.get("w_sc", {})
    d_sc = algo.get("d_sc", {})

    m_bar = progress_bar(m_sc) if m_sc else "░░░░"
    w_bar = progress_bar(w_sc) if w_sc else "░░░░"
    d_bar = progress_bar(d_sc) if d_sc else "░░░░"

    price   = round(float(last_row["close"]),     2)
    ema10   = round(float(last_row["ema10"]),      2)
    ema21   = round(float(last_row["ema21"]),      2)
    ema200  = round(float(last_row["ema200"]),     2)
    rsi_val = round(float(last_row["rsi"]),        1)
    macd_v  = round(float(last_row["macd_line"]), 3)

    rsi_zone  = "overbought" if rsi_val >= 70 \
        else "above 50 ✓" if rsi_val >= 50 else "below 50 ✗"
    macd_zone = "above zero ✓" if macd_v > 0 else "below zero ✗"

    sig_lines = "\n".join(f"  • {s['type']}" for s in signals)

    extra = ""
    if algo.get("h1_exhaustion"):
        extra = "\n⏳ <b>WAIT</b> — RSI/MACD exhaustion detected"
    elif algo.get("near_ath"):
        extra = "\n★ Near ATH — breakout zone"

    vol_line = (f"Volume: {vol_ratio}× avg "
                f"{'🔥 spike' if vol_ratio >= VOL_SPIKE_MULTIPLIER else ''}")

    return (
        f"{dir_emoji} <b>ZenPro2026 — {symbol}</b>  {dir_text}\n"
        f"{'─'*32}\n"
        f"<b>Signals fired:</b>\n{sig_lines}\n"
        f"{'─'*32}\n"
        f"{state_emoji} <b>Algo Box: {sig_state}</b>  [{total_score}/24]\n"
        f"M:{m_bar}  W:{w_bar}  D:{d_bar}\n"
        f"M×3={m_sc.get('score',0)*3}  "
        f"W×2={w_sc.get('score',0)*2}  "
        f"D×1={d_sc.get('score',0)}\n"
        f"{'─'*32}\n"
        f"Price:  ₹{price}\n"
        f"EMA10:  {ema10}  |  EMA21: {ema21}  |  EMA200: {ema200}\n"
        f"MACD:   {macd_v:+.3f} ({macd_zone})\n"
        f"RSI:    {rsi_val} ({rsi_zone})\n"
        f"{vol_line}"
        f"{extra}\n"
        f"{'─'*32}\n"
        f"<i>{now_ist}</i>\n"
        f"<i>zeninvestor.in</i>"
    )

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — COOLDOWN MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def load_cooldown() -> dict:
    try:
        if os.path.exists(COOLDOWN_FILE):
            with open(COOLDOWN_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_cooldown(state: dict):
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"  [Cooldown save error] {e}")

def is_in_cooldown(symbol: str, cooldown_state: dict) -> bool:
    last_ts = cooldown_state.get(symbol)
    if not last_ts:
        return False
    return (time.time() - last_ts) / 60 < COOLDOWN_MINUTES

def update_cooldown(symbol: str, cooldown_state: dict):
    cooldown_state[symbol] = time.time()

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — MARKET HOURS CHECK
# ═══════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    IST  = pytz.timezone("Asia/Kolkata")
    now  = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

# ═══════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ENGINE
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 50)
    print(f"ZenPro2026 Signal Engine  —  {datetime.now()}")
    print("=" * 50)

    if not is_market_open():
        print("Market closed — skipping run.")
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return

    cooldown_state = load_cooldown()
    alerts_sent    = 0

    for symbol, yahoo_symbol in WATCHLIST.items():
        print(f"\nProcessing {symbol} ({yahoo_symbol})...")

        if is_in_cooldown(symbol, cooldown_state):
            print(f"  Skipping — in cooldown ({COOLDOWN_MINUTES} min)")
            continue

        df = fetch_daily_ohlcv(yahoo_symbol, days=600)
        if df.empty or len(df) < 250:
            print(f"  Not enough data ({len(df)} bars) — skipping")
            continue

        print(f"  Fetched {len(df)} daily bars")

        signals, last_row = detect_signals(df)

        if not signals or last_row is None:
            print(f"  No signals fired")
            continue

        print(f"  Signals fired: {[s['type'] for s in signals]}")

        algo = {}
        if SIG_ALGO_BOX:
            algo = algo_signal_box(df)
            print(f"  Algo Box: {algo.get('sig_state','—')} "
                  f"[{algo.get('total_score',0)}/24]")
        else:
            algo = {"sig_state": "—", "total_score": 0,
                    "m_sc": {}, "w_sc": {}, "d_sc": {}}

        vol_ratio = round(float(last_row.get("vol_ratio", 1.0)), 2)
        message   = format_alert(symbol, signals, last_row, algo, vol_ratio)

        print(f"  Sending Telegram alert...")
        if send_telegram(message):
            print(f"  Alert sent! ✓")
            update_cooldown(symbol, cooldown_state)
            alerts_sent += 1
        else:
            print(f"  Telegram send FAILED")

        time.sleep(1.5)

    save_cooldown(cooldown_state)
    print(f"\nDone. {alerts_sent} alert(s) sent.")

if __name__ == "__main__":
    run()
