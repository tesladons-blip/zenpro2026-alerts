#!/usr/bin/env python3
"""
ZenPro2026 Signal Engine
========================
Translates your TradingView ZenPro2026 indicator into Python.
Runs via GitHub Actions every 5 min during market hours (9:15–15:30 IST).
Fetches live data from Dhan API → calculates signals → sends Telegram alerts.

HOW TO USE:
  1. Upload this file to your GitHub repo
  2. Add secrets in GitHub: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN,
     TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  3. Add schedule.yml (provided separately) to .github/workflows/
  4. Done — alerts will fire automatically during market hours
"""

import os, json, time, math, pytz
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import requests

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CREDENTIALS  (read from GitHub Secrets, never hardcode)
# ═══════════════════════════════════════════════════════════════

DHAN_CLIENT_ID    = os.environ.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — SIGNAL SWITCHES  (True = enabled, False = skip)
# These match the User Selection toggles in your TradingView indicator
# ═══════════════════════════════════════════════════════════════

SIG_MACD_ZERO_CROSS  = True    # MACD line crosses above/below ZERO
SIG_MACD_LINE_CROSS  = False   # MACD line crosses signal line
SIG_EMA10_21_CROSS   = True    # EMA 10 crosses EMA 21
SIG_RSI_MOMENTUM     = True    # RSI crosses 60 up (buy) / 50 down (sell)
SIG_RSI_50_ENTRY     = False   # RSI crosses 50
SIG_RSI_EARLY        = False   # RSI crosses its own SMA (earliest signal)
SIG_ALGO_BOX         = True    # Full weighted MTF scoring (M×3 + W×2 + D×1)
SIG_VOLUME_SPIKE     = True    # Only alert when volume > 2× 20-bar average

# Algo Box score thresholds (max score = 24)
ALGO_STRONG_BUY_MIN  = 20
ALGO_BUY_MIN         = 15
ALGO_BUY_DIP_MIN     = 10
ALGO_HOLD_MIN        = 6

VOL_SPIKE_MULTIPLIER = 2.0     # Volume must be this × average to count as spike
COOLDOWN_MINUTES     = 30      # Suppress repeat alerts per symbol (minutes)
COOLDOWN_FILE        = "cooldown_state.json"  # Persists between runs on GitHub

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — WATCHLIST
# Format: "DISPLAY_NAME": ("security_id", "exchange_segment", "instrument_type")
#
# How to find security_id:
#   Download: https://images.dhan.co/api-data/api-scrip-master.csv
#   Search for your stock name → copy the SEM_SMST_SECURITY_ID column
#
# exchange_segment options: "NSE_EQ" (NSE equity), "BSE_EQ" (BSE equity), "IDX_I" (index)
# instrument_type options:  "EQUITY", "INDEX"
# ═══════════════════════════════════════════════════════════════

WATCHLIST = {
    "RELIANCE":  ("500325",  "NSE_EQ", "EQUITY"),
    "TCS":       ("532540",  "NSE_EQ", "EQUITY"),
    "HDFCBANK":  ("1333",    "NSE_EQ", "EQUITY"),
    "INFY":      ("500209",  "NSE_EQ", "EQUITY"),
    "WIPRO":     ("507685",  "NSE_EQ", "EQUITY"),
    "NIFTY50":   ("13",      "IDX_I",  "INDEX"),
    "BANKNIFTY": ("25",      "IDX_I",  "INDEX"),
}

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DHAN API HELPERS
# ═══════════════════════════════════════════════════════════════

DHAN_BASE = "https://api.dhan.co"

def dhan_headers():
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id":    DHAN_CLIENT_ID,
        "Content-Type": "application/json",
    }

def fetch_daily_ohlcv(security_id: str, exchange_segment: str,
                      instrument_type: str, days: int = 500) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Dhan API.
    Returns DataFrame with columns: date, open, high, low, close, volume
    """
    to_dt   = date.today()
    from_dt = to_dt - timedelta(days=days)

    payload = {
        "securityId":      security_id,
        "exchangeSegment": exchange_segment,
        "instrument":      instrument_type,
        "expiryCode":      0,
        "fromDate":        from_dt.strftime("%Y-%m-%d"),
        "toDate":          to_dt.strftime("%Y-%m-%d"),
    }

    try:
        resp = requests.post(
            f"{DHAN_BASE}/v2/charts/historical",
            headers=dhan_headers(),
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or not data["data"]:
            return pd.DataFrame()

        d = data["data"]
        df = pd.DataFrame({
            "date":   pd.to_datetime(d.get("timestamp", []), unit="s"),
            "open":   d.get("open",   []),
            "high":   d.get("high",   []),
            "low":    d.get("low",    []),
            "close":  d.get("close",  []),
            "volume": d.get("volume", []),
        })
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"  [Dhan fetch error] {security_id}: {e}")
        return pd.DataFrame()

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample daily OHLCV to weekly ('W') or monthly ('M').
    """
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
# Direct Python translation of your Pine Script formulas
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
    """Returns (macd_line, signal_line, histogram)"""
    ema_fast   = calc_ema(close, fast)
    ema_slow   = calc_ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def crossover(a: pd.Series, b) -> pd.Series:
    """True where a crosses above b (was below, now above)."""
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a.shift(1) < b.shift(1)) & (a >= b)

def crossunder(a: pd.Series, b) -> pd.Series:
    """True where a crosses below b (was above, now below)."""
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a.shift(1) > b.shift(1)) & (a <= b)

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all ZenPro2026 indicators on a daily OHLCV DataFrame.
    Adds columns for every signal condition.
    """
    c = df["close"].copy()
    v = df["volume"].copy()

    # ── EMAs
    df["ema10"]  = calc_ema(c, 10)
    df["ema21"]  = calc_ema(c, 21)
    df["ema50"]  = calc_ema(c, 50)
    df["ema100"] = calc_ema(c, 100)
    df["ema150"] = calc_ema(c, 150)
    df["ema200"] = calc_ema(c, 200)
    df["sma200"] = c.rolling(200).mean()

    # EMA 13/27 for RSI baseline trend (from your rsi_baseline function)
    df["ema13"] = calc_ema(c, 13)
    df["ema27"] = calc_ema(c, 27)

    # ── RSI
    df["rsi"]     = calc_rsi(c, 14)
    df["rsi_sma"] = df["rsi"].rolling(14).mean()  # SMA of RSI (ta.sma in Pine)

    # RSI states
    df["rsi_rising"]  = df["rsi"] > df["rsi"].shift(1)
    df["rsi_falling"] = df["rsi"] < df["rsi"].shift(1)

    # ── MACD (12, 26, 9)
    df["macd_line"], df["macd_sig"], df["macd_hist"] = calc_macd(c)

    # ── Volume spike: current > N× 20-bar average
    df["vol_avg20"]   = v.rolling(20).mean()
    df["vol_spike"]   = v > (df["vol_avg20"] * VOL_SPIKE_MULTIPLIER)
    df["vol_ratio"]   = (v / df["vol_avg20"]).round(2)

    # ── ATH: within 3% of 252-bar high
    df["high252"]     = df["high"].rolling(252).max()
    df["near_ath"]    = c >= df["high252"] * 0.97

    # ── CROSS CONDITIONS (translated directly from Pine Script)
    # MACD crosses
    df["cross_macdz_up"]    = crossover(df["macd_line"], 0)
    df["cross_macdz_dn"]    = crossunder(df["macd_line"], 0)
    df["cross_macdl_up"]    = crossover(df["macd_line"], df["macd_sig"])
    df["cross_macdl_dn"]    = crossunder(df["macd_line"], df["macd_sig"])

    # EMA 10 × EMA 21 cross
    df["cross_ema10_21_up"] = crossover(df["ema10"], df["ema21"])
    df["cross_ema10_21_dn"] = crossunder(df["ema10"], df["ema21"])

    # RSI early entry (RSI crosses its own SMA)
    df["cross_rsi_sma_up"]  = crossover(df["rsi"], df["rsi_sma"])
    df["cross_rsi_sma_dn"]  = crossunder(df["rsi"], df["rsi_sma"])

    # RSI 50 entry
    df["cross_rsi_50_up"]   = crossover(df["rsi"], 50) & df["rsi_rising"]
    df["cross_rsi_50_dn"]   = crossunder(df["rsi"], 50) & df["rsi_falling"]

    # RSI momentum (cross 60 up, cross 50 down)
    df["cross_rsi_60_up"]   = crossover(df["rsi"], 60) & df["rsi_rising"]
    df["cross_rsi_50_sell"] = crossunder(df["rsi"], 50) & df["rsi_falling"]

    return df

def calc_tf_score(df_daily: pd.DataFrame, rule: str) -> dict:
    """
    Resample to weekly or monthly, calculate the 4 ladder levels,
    return the score (0–4) for that timeframe.
    Ladder: L1=RSI>RSI_SMA | L2=MACD>Signal | L3=RSI>50 | L4=EMA10>EMA21
    """
    if df_daily.empty:
        return {"l1": 0, "l2": 0, "l3": 0, "l4": 0, "score": 0}

    df_tf = resample_ohlcv(df_daily[["date","open","high","low","close","volume"]], rule)
    if len(df_tf) < 30:
        return {"l1": 0, "l2": 0, "l3": 0, "l4": 0, "score": 0}

    df_tf = calc_indicators(df_tf)
    last  = df_tf.iloc[-1]

    l1 = 1 if last["rsi"] > last["rsi_sma"] else 0
    l2 = 1 if last["macd_line"] > last["macd_sig"] else 0
    l3 = 1 if last["rsi"] > 50 else 0
    l4 = 1 if last["ema10"] > last["ema21"] else 0

    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "score": l1+l2+l3+l4}

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — ALGO SIGNAL BOX
# Weighted MTF score: Monthly×3 + Weekly×2 + Daily×1 = max 24
# ═══════════════════════════════════════════════════════════════

def progress_bar(sc: dict) -> str:
    """Render the ████░░ style bar from your indicator."""
    return "".join("█" if sc[k] else "░" for k in ["l1","l2","l3","l4"])

def algo_signal_box(df_daily: pd.DataFrame) -> dict:
    """
    Full ZenPro2026 Algo Signal Box calculation.
    Returns dict with sig_state, total_score, breakdown, exhaustion flags.
    """
    if df_daily.empty or len(df_daily) < 250:
        return {"sig_state": "NO DATA", "total_score": 0}

    # Daily score
    d_sc = calc_tf_score(df_daily, rule="D")
    # Actually for daily just use the df_daily directly
    df_i = calc_indicators(df_daily.copy())
    last_d = df_i.iloc[-1]
    d_sc = {
        "l1": 1 if last_d["rsi"] > last_d["rsi_sma"] else 0,
        "l2": 1 if last_d["macd_line"] > last_d["macd_sig"] else 0,
        "l3": 1 if last_d["rsi"] > 50 else 0,
        "l4": 1 if last_d["ema10"] > last_d["ema21"] else 0,
        "score": 0,
    }
    d_sc["score"] = d_sc["l1"] + d_sc["l2"] + d_sc["l3"] + d_sc["l4"]

    # Weekly score
    w_sc = calc_tf_score(df_daily, "W")

    # Monthly score
    m_sc = calc_tf_score(df_daily, "ME")  # "ME" = month-end in pandas

    total_score = (m_sc["score"] * 3) + (w_sc["score"] * 2) + (d_sc["score"] * 1)

    # ATH upgrade
    near_ath = bool(last_d["near_ath"])
    if near_ath and total_score >= ALGO_BUY_MIN:
        eff_score = max(total_score, ALGO_STRONG_BUY_MIN)
    else:
        eff_score = total_score

    # SMA 200 structural check
    above_d200 = last_d["close"] > last_d["sma200"] if not math.isnan(last_d["sma200"]) else True

    # Weekly SMA 200
    df_w = resample_ohlcv(df_daily[["date","open","high","low","close","volume"]], "W")
    df_w["sma200"] = df_w["close"].rolling(200).mean()
    above_w200 = True
    if len(df_w) >= 200:
        above_w200 = float(df_w["close"].iloc[-1]) > float(df_w["sma200"].iloc[-1])

    trend_down = not above_d200 and not above_w200
    trend_weak = not above_d200 and above_w200

    # 1H exhaustion (we approximate using the last bar's daily data for now)
    # In a GitHub Actions context, getting 1H data would require a separate fetch
    # We flag if daily RSI ≥ 70 as a caution signal
    h1_rsi_ob   = last_d["rsi"] >= 70
    h1_macd_exh = (last_d["macd_line"] > 0) and (last_d["macd_line"] < last_d["macd_sig"])
    h1_exhaustion = h1_rsi_ob or h1_macd_exh

    # Determine signal state
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
        "sig_state":    sig_state,
        "total_score":  total_score,
        "eff_score":    eff_score,
        "m_sc":         m_sc,
        "w_sc":         w_sc,
        "d_sc":         d_sc,
        "near_ath":     near_ath,
        "h1_rsi_ob":    h1_rsi_ob,
        "h1_macd_exh":  h1_macd_exh,
        "h1_exhaustion": h1_exhaustion,
        "trend_down":   trend_down,
        "trend_weak":   trend_weak,
    }

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — SIGNAL DETECTION (per bar, daily timeframe)
# Returns list of fired signals with type + direction
# ═══════════════════════════════════════════════════════════════

def detect_signals(df: pd.DataFrame) -> list:
    """
    Run all enabled signal checks on the latest candle.
    Returns list of dicts: [{"type": "...", "direction": "BUY"/"SELL"}, ...]
    """
    if df.empty or len(df) < 30:
        return []

    df = calc_indicators(df)
    last = df.iloc[-1]
    fired = []

    # MACD Zero Cross
    if SIG_MACD_ZERO_CROSS:
        if last["cross_macdz_up"]:
            fired.append({"type": "MACD Zero Cross UP",  "direction": "BUY"})
        if last["cross_macdz_dn"]:
            fired.append({"type": "MACD Zero Cross DN", "direction": "SELL"})

    # MACD Line Cross
    if SIG_MACD_LINE_CROSS:
        if last["cross_macdl_up"]:
            fired.append({"type": "MACD Line Cross UP",  "direction": "BUY"})
        if last["cross_macdl_dn"]:
            fired.append({"type": "MACD Line Cross DN", "direction": "SELL"})

    # EMA 10 × EMA 21
    if SIG_EMA10_21_CROSS:
        if last["cross_ema10_21_up"]:
            fired.append({"type": "EMA10 x EMA21 UP",  "direction": "BUY"})
        if last["cross_ema10_21_dn"]:
            fired.append({"type": "EMA10 x EMA21 DN", "direction": "SELL"})

    # RSI Momentum (cross 60 up / cross 50 dn)
    if SIG_RSI_MOMENTUM:
        if last["cross_rsi_60_up"]:
            fired.append({"type": "RSI Momentum BUY (crossed 60)", "direction": "BUY"})
        if last["cross_rsi_50_sell"]:
            fired.append({"type": "RSI Momentum SELL (crossed 50)", "direction": "SELL"})

    # RSI 50 Entry
    if SIG_RSI_50_ENTRY:
        if last["cross_rsi_50_up"]:
            fired.append({"type": "RSI 50 Entry BUY",  "direction": "BUY"})
        if last["cross_rsi_50_dn"]:
            fired.append({"type": "RSI 50 Entry SELL", "direction": "SELL"})

    # RSI Early (RSI crosses its own SMA)
    if SIG_RSI_EARLY:
        if last["cross_rsi_sma_up"]:
            fired.append({"type": "RSI Early Entry BUY",  "direction": "BUY"})
        if last["cross_rsi_sma_dn"]:
            fired.append({"type": "RSI Early Exit SELL", "direction": "SELL"})

    # Volume filter: if enabled, drop signals where no volume spike
    if SIG_VOLUME_SPIKE and fired:
        if not last["vol_spike"]:
            fired = []  # suppress all signals — no confirmation

    return fired, last

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — TELEGRAM MESSAGING
# ═══════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """Send a message to your Telegram group via Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
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
    """
    Build the Telegram message for your group.
    Matches the style of the ZenPro2026 Signal Box.
    """
    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST).strftime("%d %b %Y  %H:%M IST")

    # Direction emoji
    directions = set(s["direction"] for s in signals)
    if "BUY" in directions and "SELL" not in directions:
        dir_emoji = "🟢"
        dir_text  = "BUY"
    elif "SELL" in directions and "BUY" not in directions:
        dir_emoji = "🔴"
        dir_text  = "SELL / EXIT"
    else:
        dir_emoji = "🟡"
        dir_text  = "MIXED"

    # Signal box state emoji
    state_emoji = {
        "STRONG BUY":  "⬆",
        "BUY":         "↑",
        "BUY DIP":     "↗",
        "HOLD":        "◆",
        "CAUTION":     "⚠",
        "TREND WEAK":  "⚠",
        "TREND DOWN":  "↓",
        "NO DATA":     "—",
    }.get(algo.get("sig_state", ""), "—")

    sig_state   = algo.get("sig_state", "—")
    total_score = algo.get("total_score", 0)
    m_sc        = algo.get("m_sc", {})
    w_sc        = algo.get("w_sc", {})
    d_sc        = algo.get("d_sc", {})

    m_bar = progress_bar(m_sc) if m_sc else "░░░░"
    w_bar = progress_bar(w_sc) if w_sc else "░░░░"
    d_bar = progress_bar(d_sc) if d_sc else "░░░░"

    # Indicator values
    price   = round(float(last_row["close"]), 2)
    ema10   = round(float(last_row["ema10"]),  2)
    ema21   = round(float(last_row["ema21"]),  2)
    ema200  = round(float(last_row["ema200"]), 2)
    rsi_val = round(float(last_row["rsi"]),    1)
    macd_v  = round(float(last_row["macd_line"]), 3)

    rsi_zone = "overbought" if rsi_val >= 70 else "above 50 ✓" if rsi_val >= 50 else "below 50 ✗"
    macd_zone = "above zero ✓" if macd_v > 0 else "below zero ✗"

    # Signal lines fired
    sig_lines = "\n".join(f"  • {s['type']}" for s in signals)

    # Exhaustion / ATH note
    extra = ""
    if algo.get("h1_exhaustion"):
        extra = "\n⏳ <b>WAIT</b> — RSI/MACD exhaustion detected"
    elif algo.get("near_ath"):
        extra = "\n★ Near ATH — breakout zone"

    vol_line = f"Volume: {vol_ratio}× avg {'🔥 spike' if vol_ratio >= VOL_SPIKE_MULTIPLIER else ''}"

    msg = (
        f"{dir_emoji} <b>ZenPro2026 — {symbol}</b>  {dir_text}\n"
        f"{'─'*32}\n"
        f"<b>Signals fired:</b>\n{sig_lines}\n"
        f"{'─'*32}\n"
        f"{state_emoji} <b>Algo Box: {sig_state}</b>  [{total_score}/24]\n"
        f"M:{m_bar}  W:{w_bar}  D:{d_bar}\n"
        f"M×3={m_sc.get('score',0)*3}  W×2={w_sc.get('score',0)*2}  D×1={d_sc.get('score',0)}\n"
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
    return msg

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — COOLDOWN MANAGEMENT
# Prevents the same signal spamming your group every 5 minutes
# ═══════════════════════════════════════════════════════════════

def load_cooldown() -> dict:
    """Load last-alert timestamps from file."""
    try:
        if os.path.exists(COOLDOWN_FILE):
            with open(COOLDOWN_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_cooldown(state: dict):
    """Save last-alert timestamps to file."""
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"  [Cooldown save error] {e}")

def is_in_cooldown(symbol: str, cooldown_state: dict) -> bool:
    """Return True if this symbol was alerted within COOLDOWN_MINUTES."""
    last_ts = cooldown_state.get(symbol)
    if not last_ts:
        return False
    elapsed = (time.time() - last_ts) / 60
    return elapsed < COOLDOWN_MINUTES

def update_cooldown(symbol: str, cooldown_state: dict):
    cooldown_state[symbol] = time.time()

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — MARKET HOURS CHECK
# ═══════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    """Return True only during NSE trading hours (9:15–15:30 IST, Mon–Fri)."""
    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)
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

    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        print("ERROR: DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN not set in environment.")
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in environment.")
        return

    cooldown_state = load_cooldown()
    alerts_sent    = 0

    for symbol, (sec_id, exch_seg, instr_type) in WATCHLIST.items():
        print(f"\nProcessing {symbol}...")

        # Cooldown check
        if is_in_cooldown(symbol, cooldown_state):
            print(f"  Skipping — in cooldown ({COOLDOWN_MINUTES} min)")
            continue

        # Fetch data (500 trading days = ~2 years, enough for EMA 200 + MTF)
        df = fetch_daily_ohlcv(sec_id, exch_seg, instr_type, days=600)
        if df.empty or len(df) < 250:
            print(f"  Not enough data ({len(df)} bars) — skipping")
            continue

        print(f"  Fetched {len(df)} daily bars")

        # Calculate indicators and detect signals
        result = detect_signals(df)
        if result is None:
            continue
        signals, last_row = result

        if not signals:
            print(f"  No signals fired")
            continue

        print(f"  Signals fired: {[s['type'] for s in signals]}")

        # Algo Signal Box (runs even if signals already found — for score display)
        algo = {}
        if SIG_ALGO_BOX:
            algo = algo_signal_box(df)
            print(f"  Algo Box: {algo.get('sig_state','—')}  [{algo.get('total_score',0)}/24]")
        else:
            algo = {"sig_state": "—", "total_score": 0, "m_sc": {}, "w_sc": {}, "d_sc": {}}

        # Build and send message
        vol_ratio = round(float(last_row.get("vol_ratio", 1.0)), 2)
        message   = format_alert(symbol, signals, last_row, algo, vol_ratio)

        print(f"  Sending Telegram alert...")
        if send_telegram(message):
            print(f"  Alert sent!")
            update_cooldown(symbol, cooldown_state)
            alerts_sent += 1
        else:
            print(f"  Telegram send FAILED")

        # Small delay between symbols to avoid API rate limits
        time.sleep(1.5)

    save_cooldown(cooldown_state)
    print(f"\nDone. {alerts_sent} alert(s) sent.")

if __name__ == "__main__":
    run()
