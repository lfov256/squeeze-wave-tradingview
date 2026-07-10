#!/usr/bin/env python3
"""
Squeeze Wave Daily Alerts - Con registro automático de señales
"""

import json
import os
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks

DATA_API_KEY = os.getenv("POLYGON_API_KEY")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RECIPIENT_EMAIL = os.getenv("ALERT_EMAIL")

SUBS_FILE = Path("subscriptions.json")
SIGNALS_FILE = Path("data/signals_history.csv")

ASSETS = {
    "SPY (S&P 500)": "SPY",
    "QQQ (Nasdaq)": "QQQ",
    "Oro (XAU/USD)": "C:XAUUSD",
    "Plata (XAG/USD)": "C:XAGUSD",
    "BTC/USD": "X:BTCUSD",
    "ETH/USD": "X:ETHUSD",
    "EUR/USD": "C:EURUSD",
    "Libra Dólar (GBP/USD)": "C:GBPUSD",
    "Yen Dólar (USD/JPY)": "C:USDJPY",
    "Petróleo (USO)": "USO",
}

ALERT_MIN_SI = 75
ALERT_MIN_STRENGTH = 0.60
SCAN_DAYS = 120
PARAMS = {"window": 20, "bb_mult": 2.0, "kc_mult": 1.5, "atr_period": 20,
          "threshold": 0.15, "use_spectrum": True, "use_vol_filter": True}

UTC = timezone.utc

# ==================== SIGNAL LOGGING ====================
def ensure_signals_file():
    if not SIGNALS_FILE.exists():
        SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SIGNALS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "ticker", "direction", "squeeze_index",
                "signal_strength", "squeeze_detected", "trend", "lambda", "price", "priority"
            ])

def log_signal(last, ticker, priority="MEDIA"):
    ensure_signals_file()
    with open(SIGNALS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(UTC).isoformat(),
            ticker,
            last.get("Direction", ""),
            round(last.get("SqueezeIndex", 0), 1),
            round(last.get("SignalStrength", 0), 3),
            last.get("SqueezeDetected", False),
            round(last.get("Trend", 0), 3),
            round(last.get("Lambda", 0), 1),
            round(last.get("Close", 0), 4),
            priority
        ])

# ==================== CORE FUNCTIONS ====================
def fetch_data(ticker, days=365):
    now = datetime.now(UTC)
    end_date = (now + timedelta(days=1)).date()
    start_date = end_date - timedelta(days=days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000&apiKey={DATA_API_KEY}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "results" not in data or not data.get("results"):
            return pd.DataFrame()
        df = pd.DataFrame(data["results"])
        df["Date"] = pd.to_datetime(df["t"], unit="ms").dt.date
        df["Open"] = df.get("o", df["c"])
        df["High"] = df["h"]
        df["Low"] = df["l"]
        df["Close"] = df["c"]
        df["Volume"] = df.get("v", 0)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return pd.DataFrame()

def compute_atr(df, period):
    prev_c = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1).fillna(0)
    return tr.ewm(span=period, adjust=False).mean()

def compute_lambda_vectorized(smoothed, window, use_spectrum):
    n = len(smoothed)
    lambda_arr = np.full(n, np.nan)
    for i in range(window-1, n):
        seg = smoothed.values[i-window+1:i+1]
        if use_spectrum and len(seg) >= 16:
            try:
                freqs, psd = welch(seg, nperseg=len(seg))
                min_freq = 2.0 / window
                valid = freqs > min_freq
                if valid.sum() > 1:
                    dom_freq = freqs[valid][np.argmax(psd[valid])]
                    lambda_arr[i] = 1.0 / dom_freq
                else:
                    lambda_arr[i] = window / 2
            except:
                lambda_arr[i] = window / 2
        else:
            peaks, _ = find_peaks(seg)
            valleys, _ = find_peaks(-seg)
            extrema = np.sort(np.concatenate([peaks, valleys]))
            lambda_arr[i] = float(np.mean(np.diff(extrema))) if len(extrema) > 1 else window / 2
    return pd.Series(lambda_arr, index=smoothed.index).ffill().bfill().clip(lower=2.0)

def compute_trend(df, smoothed, lam):
    n = len(df)
    prices_arr = smoothed.values
    atr_arr = df.get("ATR", pd.Series([0.0] * n)).values
    lam_arr = lam.values
    slope_long_arr = np.zeros(n)
    slope_short_arr = np.zeros(n)
    for i in range(n):
        lv = max(6, int(lam_arr[i]))
        if i >= lv - 1 and atr_arr[i] > 0:
            seg = prices_arr[i - lv + 1 : i + 1]
            if len(seg) >= 2:
                try:
                    slope_long_arr[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0] / atr_arr[i]
                except:
                    pass
        short_w = 6
        if i >= short_w - 1 and atr_arr[i] > 0:
            seg_s = prices_arr[i - short_w + 1 : i + 1]
            if len(seg_s) == short_w:
                try:
                    slope_short_arr[i] = np.polyfit(np.arange(short_w), seg_s, 1)[0] / atr_arr[i]
                except:
                    pass
    slope_long_norm = np.tanh(slope_long_arr * 5)
    slope_short_norm = np.tanh(slope_short_arr * 5)
    mid = (df["High"] + df["Low"]) / 2
    range_ = (df["High"] - df["Low"]).replace(0, np.nan)
    bp = ((df["Close"] - mid) / range_).clip(-1, 1).fillna(0)
    raw_flow = bp * range_.fillna(0)
    pos_flow = raw_flow.clip(lower=0).rolling(14).sum()
    neg_flow = (-raw_flow).clip(lower=0).rolling(14).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        mfi_raw = np.where(neg_flow != 0, 100 - 100 / (1 + pos_flow / neg_flow), 50.0)
    mfi_score = ((pd.Series(mfi_raw, index=df.index) - 50) / 50).clip(-1, 1)
    roc = df["Close"].pct_change(5).fillna(0)
    roc_normalized = np.tanh(roc / (df["ATR"] / df["Close"].replace(0, np.nan)).fillna(0.01) * 3)
    trend = (0.35 * pd.Series(slope_long_norm, index=df.index) + 0.25 * pd.Series(slope_short_norm, index=df.index) + 0.25 * mfi_score + 0.15 * roc_normalized).ewm(span=2, adjust=False).mean()
    return trend

def detect_squeeze_episodes(squeeze_on, min_duration=3):
    episode = np.zeros(len(squeeze_on), dtype=int)
    ep_id = 0
    in_sq = False
    start = 0
    for i, val in enumerate(squeeze_on.values):
        if val and not in_sq:
            in_sq = True
            start = i
        elif not val and in_sq:
            in_sq = False
            if i - start >= min_duration:
                ep_id += 1
                episode[start:i] = ep_id
        elif val and in_sq and i == len(squeeze_on) - 1:
            if i - start + 1 >= min_duration:
                ep_id += 1
                episode[start:] = ep_id
    return pd.Series(episode, index=squeeze_on.index)

def calculate_squeeze_index(df, window=20, bb_mult=2.0, kc_mult=1.5, atr_period=20, threshold=0.15, use_spectrum=True, use_vol_filter=True):
    if df.empty or len(df) < window + 10: return df
    df = df.copy()
    df["EMA"] = df["Close"].ewm(span=window, adjust=False).mean()
    df["STD"] = df["Close"].rolling(window).std()
    df["UpperBB"] = df["EMA"] + bb_mult * df["STD"]
    df["LowerBB"] = df["EMA"] - bb_mult * df["STD"]
    df["BBWidth"] = (df["UpperBB"] - df["LowerBB"]) / df["EMA"].replace(0, np.nan)
    df["ATR"] = compute_atr(df, atr_period)
    df["UpperKC"] = df["EMA"] + kc_mult * df["ATR"]
    df["LowerKC"] = df["EMA"] - kc_mult * df["ATR"]
    smoothed = df["Close"].ewm(span=5, adjust=False).mean()
    df["Lambda"] = compute_lambda_vectorized(smoothed, window, use_spectrum)
    bb_percentile = df["BBWidth"].rolling(window * 3).rank(pct=True).fillna(0.5)
    compression_factor = (1 - bb_percentile).clip(0, 1)
    lambda_quality = 1 / (1 + ((df["Lambda"] - window / 3) / (window / 4)) ** 2)
    raw_si = compression_factor / df["BBWidth"].replace(0, np.nan)
    si_roll_max = raw_si.rolling(window * 3).max().replace(0, np.nan)
    df["SqueezeIndex"] = ((raw_si / si_roll_max) * 100 * lambda_quality).clip(0, 100).fillna(0)
    df["SqueezeOn"] = (df["UpperBB"] <= df["UpperKC"]) & (df["LowerBB"] >= df["LowerKC"])
    df["SqueezeEpisode"] = detect_squeeze_episodes(df["SqueezeOn"])
    df["Trend"] = compute_trend(df, smoothed, df["Lambda"])
    df["Direction"] = np.where(df["Trend"] > 0, "Alcista", np.where(df["Trend"] < 0, "Bajista", "Neutral"))
    df["SqueezeDetected"] = df["SqueezeOn"] & (df["Trend"].abs() > threshold)
    if use_vol_filter:
        vol_pct = df["ATR"].rolling(50).rank(pct=True).fillna(0.5)
        df["SqueezeDetected"] = df["SqueezeDetected"] & (vol_pct < 0.75)
    df["SignalStrength"] = (0.6 * df["SqueezeIndex"] / 100 + 0.4 * df["Trend"].abs().clip(0, 1)).clip(0, 1)
    return df

def send_email_resend(to_email, subject, body):
    if not RESEND_API_KEY:
        print(body)
        return False
    try:
        r = requests.get("https://api.resend.com/emails", headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"}, json={"from": "Squeeze Alerts <alerts@resend.dev>", "to": [to_email], "subject": subject, "text": body}, timeout=15)
        return r.status_code == 200
    except:
        return False

def main():
    today = datetime.now(UTC).weekday()
    if today in (0, 5, 6):
        print("Sin alertas hoy")
        return
    subs = []
    if SUBS_FILE.exists():
        try: subs = json.loads(SUBS_FILE.read_text())
        except: pass
    if not subs: subs = list(ASSETS.keys())
    active = []
    for name in subs:
        ticker = ASSETS.get(name, name)
        df = fetch_data(ticker, SCAN_DAYS)
        if df.empty or len(df) < 40: continue
        df = calculate_squeeze_index(df, **PARAMS)
        if len(df) < 10: continue
        last = df.iloc[-1]
        si = last.get("SqueezeIndex", 0)
        strength = last.get("SignalStrength", 0)
        detected = last.get("SqueezeDetected", False)
        # Condición: ANY of the 3
        if detected or si >= ALERT_MIN_SI or strength >= ALERT_MIN_STRENGTH:
            priority = "ALTA" if si >= 90 else "MEDIA"
            icon = "🔴" if last["Direction"] == "Bajista" else "🔵"
            body = f"SQUEEZE ALERT - {name}\n{icon} {last['Direction']}\nSqueezeIndex: {si:.1f} | SignalStrength: {strength:.0%}\nTrend: {last.get('Trend', 0):+.3f} | Lambda: {last.get('Lambda', 0):.1f}d | Precio: {last.get('Close', 0):.4f}"
            send_email_resend(RECIPIENT_EMAIL, f"Squeeze Alert {name}", body)
            log_signal(last, name, priority)
            active.append(name)
    print("Enviadas:", active if active else "Ninguna")

if __name__ == "__main__":
    main()