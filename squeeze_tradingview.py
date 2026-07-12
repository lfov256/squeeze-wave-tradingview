"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SQUEEZE INDEX                                    ║    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import linregress
from datetime import datetime, timedelta, timezone
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

UTC = timezone.utc

st.set_page_config(
    page_title="SqueezeIndex",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c10;
  }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.03em; }

  /* KPI Cards */
  .kpi-card {
    background: #0d1117;
    border: 1px solid #1e2630;
    border-radius: 12px;
    padding: 20px 16px 14px 16px;
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .kpi-card.green::before { background: #00d68f; }
  .kpi-card.yellow::before { background: #ffc107; }
  .kpi-card.red::before { background: #ff4757; }
  .kpi-card.blue::before { background: #4da6ff; }
  .kpi-card.purple::before { background: #b48eff; }

  .kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7685;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .kpi-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1;
    margin-bottom: 6px;
  }
  .kpi-sub {
    font-size: 11px;
    color: #6b7685;
    font-family: 'IBM Plex Mono', monospace;
  }
  .kpi-green { color: #00d68f; }
  .kpi-yellow { color: #ffc107; }
  .kpi-red { color: #ff4757; }
  .kpi-blue { color: #4da6ff; }
  .kpi-purple { color: #b48eff; }
  .kpi-white { color: #e8edf3; }

  /* Signal Banner */
  .signal-banner {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: 0.02em;
  }
  .signal-active { background: rgba(0,214,143,0.12); border: 1px solid rgba(0,214,143,0.35); color: #00d68f; }
  .signal-pending { background: rgba(255,193,7,0.10); border: 1px solid rgba(255,193,7,0.30); color: #ffc107; }
  .signal-none { background: rgba(107,118,133,0.10); border: 1px solid rgba(107,118,133,0.20); color: #6b7685; }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4da6ff;
    border-bottom: 1px solid #1e2630;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
  }

  /* Explanation boxes */
  .explain-box {
    background: #0d1117;
    border: 1px solid #1e2630;
    border-left: 3px solid #4da6ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #8b98a8;
    line-height: 1.6;
    margin: 8px 0 16px 0;
  }

  /* Backtest stat row */
  .bt-stat {
    display: inline-flex;
    flex-direction: column;
    background: #0d1117;
    border: 1px solid #1e2630;
    border-radius: 8px;
    padding: 12px 16px;
    min-width: 100px;
  }

  /* Override Streamlit defaults */
  .stButton button {
    background: #0d6efd !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
  }
  [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
  div[data-testid="stExpander"] { border: 1px solid #1e2630 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("# 〰️ SqueezeIndex v3.0")
    st.caption("Física de Ondas · Análisis Espectral · Detección de Compresión de Volatilidad")
with col_time:
    st.markdown(f"""
    <div style='text-align:right; padding-top:12px; font-family:IBM Plex Mono,monospace; font-size:11px; color:#6b7685;'>
    {datetime.now(UTC).strftime('%Y-%m-%d')}<br>
    {datetime.now(UTC).strftime('%H:%M:%S UTC')}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛️ Parámetros")

    ASSETS = {
        "EUR/USD": "C:EURUSD", "GBP/USD": "C:GBPUSD",
        "USD/JPY": "C:USDJPY", "Oro (XAU/USD)": "C:XAUUSD",
        "Plata (XAG/USD)": "C:XAGUSD", "SPY (S&P 500)": "SPY",
        "BTC/USD": "X:BTCUSD", "ETH/USD": "X:ETHUSD",
        "USO (Petróleo)": "USO", "QQQ (Nasdaq)": "QQQ",
    }
    selected_asset = st.selectbox("Activo", list(ASSETS.keys()))
    ticker = ASSETS[selected_asset]
    days = st.slider("Días de histórico", 60, 730, 365)

    st.divider()
    st.markdown("**Bandas & Canales**")
    window = st.slider("Ventana (BB / EMA)", 10, 60, 20)
    bb_mult = st.slider("Multiplicador BB", 1.0, 3.5, 2.0, 0.1,
                        help="Mayor → menos squeezes detectados (más conservador)")
    kc_mult = st.slider("Multiplicador Keltner", 1.0, 3.0, 1.5, 0.1,
                        help="Mayor → más fácil que BB entre en KC (más señales)")
    atr_period = st.slider("Período ATR", 10, 40, 20,
                           help="Volatilidad base para Keltner y normalización de Trend")

    st.markdown("**Señal**")
    threshold = st.slider("Mínimo de Trend para señal", 0.05, 0.5, 0.15, 0.01,
                          help="Cuánta dirección mínima se necesita para activar una señal fuerte")

    st.markdown("**Opciones avanzadas**")
    use_spectrum = st.checkbox("Lambda por análisis espectral (Welch)", value=True,
                               help="Más robusto que contar picos. Requiere ≥16 barras en ventana.")
    use_vol_filter = st.checkbox("Filtrar señales en alta volatilidad", value=True,
                                 help="Desactiva señales cuando ATR > percentil 75. Más selectivo.")

    st.divider()
    update = st.button("▶ Calcular", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════════════
API_KEY = st.secrets["API_KEY"]

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES CORE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_data(ticker: str, days: int = 365) -> pd.DataFrame:
    now = datetime.now(UTC)
    end_date = (now + timedelta(days=1)).date()
    start_date = end_date - timedelta(days=days)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
        f"/{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000&apiKey={API_KEY}"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        if "results" not in data or not data["results"]:
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
        st.error(f"Error descargando {ticker}: {str(e)[:120]}")
        return pd.DataFrame()


def compute_lambda_vectorized(smoothed, window, use_spectrum):
    n = len(smoothed)
    lambda_arr = np.full(n, np.nan)
    prices_arr = smoothed.values

    for i in range(window - 1, n):
        seg = prices_arr[i - window + 1: i + 1]
        if use_spectrum and len(seg) >= 16:
            try:
                # ✅ FIX: usar len(seg) como nperseg, no 8
                # Más puntos = más bins de frecuencia = resolución real
                nperseg = len(seg)  # era: min(len(seg), 8)
                freqs, psd = welch(seg, nperseg=nperseg)
                
                # Ignorar DC (índice 0) y frecuencias muy bajas (< 1/window)
                # para evitar que ruido lento domine siempre
                min_freq = 2.0 / window
                valid = freqs > min_freq
                if valid.sum() > 1:
                    dom_freq = freqs[valid][np.argmax(psd[valid])]
                    lambda_arr[i] = 1.0 / dom_freq
                else:
                    lambda_arr[i] = window / 2
            except Exception:
                lambda_arr[i] = window / 2
        else:
            peaks, _ = find_peaks(seg)
            valleys, _ = find_peaks(-seg)
            extrema = np.sort(np.concatenate([peaks, valleys]))
            if len(extrema) > 1:
                lambda_arr[i] = float(np.mean(np.diff(extrema)))
            else:
                lambda_arr[i] = window / 2

    lam = pd.Series(lambda_arr, index=smoothed.index)
    return lam.ffill().bfill().clip(lower=2.0)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_c = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"] - prev_c).abs()
    ], axis=1).max(axis=1).fillna(0)
    return tr.ewm(span=period, adjust=False).mean()


def compute_trend(df: pd.DataFrame, smoothed: pd.Series, lam: pd.Series) -> pd.Series:
    n = len(df)
    prices_arr = smoothed.values
    atr_arr = df["ATR"].values
    lam_arr = lam.values
    slope_long_arr = np.zeros(n)
    slope_short_arr = np.zeros(n)
    for i in range(n):
        lv = max(6, int(lam_arr[i]))
        atr_v = atr_arr[i]
        if atr_v == 0 or np.isnan(atr_v):
            continue
        if i >= lv - 1:
            seg = prices_arr[i - lv + 1: i + 1]
            x = np.arange(len(seg), dtype=float)
            slope_long_arr[i] = np.polyfit(x, seg, 1)[0] / atr_v
        short_w = 6
        if i >= short_w - 1:
            seg_s = prices_arr[i - short_w + 1: i + 1]
            x_s = np.arange(short_w, dtype=float)
            slope_short_arr[i] = np.polyfit(x_s, seg_s, 1)[0] / atr_v

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

    trend = (
        0.35 * pd.Series(slope_long_norm, index=df.index) +
        0.25 * pd.Series(slope_short_norm, index=df.index) +
        0.25 * mfi_score +
        0.15 * roc_normalized
    ).ewm(span=2, adjust=False).mean()
    return trend


def detect_squeeze_episodes(squeeze_on: pd.Series, min_duration: int = 3) -> pd.Series:
    episode = np.zeros(len(squeeze_on), dtype=int)
    ep_id = 0
    in_sq = False
    start = 0
    vals = squeeze_on.values
    for i, val in enumerate(vals):
        if val and not in_sq:
            in_sq = True
            start = i
        elif not val and in_sq:
            in_sq = False
            if i - start >= min_duration:
                ep_id += 1
                episode[start:i] = ep_id
        elif val and in_sq and i == len(vals) - 1:
            if i - start + 1 >= min_duration:
                ep_id += 1
                episode[start:] = ep_id
    return pd.Series(episode, index=squeeze_on.index)


def calculate_squeeze_index(
    df: pd.DataFrame, window: int, bb_mult: float, kc_mult: float,
    atr_period: int, threshold: float,
    use_spectrum: bool = True, use_vol_filter: bool = True
) -> pd.DataFrame:
    if df.empty or len(df) < window + 10:
        return df

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
    df["SqueezeEpisode"] = detect_squeeze_episodes(df["SqueezeOn"], min_duration=3)

    df["Trend"] = compute_trend(df, smoothed, df["Lambda"])
    df["Direction"] = np.where(df["Trend"] > 0, "Alcista",
                       np.where(df["Trend"] < 0, "Bajista", "Neutral"))

    df["SqueezeDetected"] = df["SqueezeOn"] & (df["Trend"].abs() > threshold)
    if use_vol_filter:
        vol_pct = df["ATR"].rolling(50).rank(pct=True).fillna(0.5)
        df["SqueezeDetected"] = df["SqueezeDetected"] & (vol_pct < 0.75)

    df["SignalStrength"] = (
        0.6 * df["SqueezeIndex"] / 100 +
        0.4 * df["Trend"].abs().clip(0, 1)
    ).clip(0, 1)

    return df


# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "green": "#00d68f",
    "red": "#ff4757",
    "yellow": "#ffc107",
    "blue": "#4da6ff",
    "purple": "#b48eff",
    "bg": "#080c10",
    "panel": "#0d1117",
    "border": "#1e2630",
    "text_dim": "#6b7685",
    "text": "#c8d0db",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="IBM Plex Mono, monospace", size=11, color=COLORS["text"]),
        xaxis=dict(gridcolor="#1a2030", showgrid=True, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1a2030", showgrid=True, zeroline=False, tickfont=dict(size=10)),
    )
)


def build_main_chart(df: pd.DataFrame, asset_name: str) -> go.Figure:
    """
    4 paneles:
    1. Precio + BB + Keltner + marcadores squeeze
    2. SqueezeIndex (gauge horizontal temporal)
    3. Trend compuesto (4 componentes)
    4. Lambda — longitud de onda dominante
    """
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.48, 0.18, 0.18, 0.16],
        vertical_spacing=0.025,
    )

    # ─ Panel 1: Precio ─────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Precio",
        increasing=dict(line=dict(color=COLORS["green"], width=1), fillcolor=COLORS["green"]),
        decreasing=dict(line=dict(color=COLORS["red"], width=1), fillcolor=COLORS["red"]),
        whiskerwidth=0.4,
    ), row=1, col=1)

    # BB
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["UpperBB"],
        line=dict(color=COLORS["blue"], width=1.2),
        name="BB Superior", showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["LowerBB"],
        line=dict(color=COLORS["blue"], width=1.2),
        fill="tonexty", fillcolor="rgba(77,166,255,0.05)",
        name="BB Inferior", showlegend=True
    ), row=1, col=1)

    # Keltner
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["UpperKC"],
        line=dict(color=COLORS["yellow"], width=1, dash="dot"),
        name="KC Superior", showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["LowerKC"],
        line=dict(color=COLORS["yellow"], width=1, dash="dot"),
        name="KC Inferior", showlegend=True
    ), row=1, col=1)

    # Fondo squeeze
    squeeze_regions = []
    in_sq = False
    for i in range(len(df)):
        if df["SqueezeOn"].iloc[i] and not in_sq:
            in_sq = True
            sq_start = df["Date"].iloc[i]
        elif not df["SqueezeOn"].iloc[i] and in_sq:
            in_sq = False
            squeeze_regions.append((sq_start, df["Date"].iloc[i - 1]))
    if in_sq:
        squeeze_regions.append((sq_start, df["Date"].iloc[-1]))

    for x0, x1 in squeeze_regions:
        fig.add_vrect(x0=x0, x1=x1,
                      fillcolor="rgba(0,214,143,0.07)", layer="below", line_width=0,
                      row=1, col=1)

    # Marcadores señal fuerte
    sq_det = df[df["SqueezeDetected"]]
    if len(sq_det) > 0:
        sq_bull = sq_det[sq_det["Direction"] == "Alcista"]
        sq_bear = sq_det[sq_det["Direction"] == "Bajista"]
        if len(sq_bull) > 0:
            fig.add_trace(go.Scatter(
                x=sq_bull["Date"], y=sq_bull["Low"] * 0.9975,
                mode="markers", name="Señal Alcista",
                marker=dict(symbol="triangle-up", color=COLORS["green"], size=9,
                            line=dict(color="#ffffff", width=0.5))
            ), row=1, col=1)
        if len(sq_bear) > 0:
            fig.add_trace(go.Scatter(
                x=sq_bear["Date"], y=sq_bear["High"] * 1.0025,
                mode="markers", name="Señal Bajista",
                marker=dict(symbol="triangle-down", color=COLORS["red"], size=9,
                            line=dict(color="#ffffff", width=0.5))
            ), row=1, col=1)

    # EMA
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["EMA"],
        line=dict(color="rgba(255,255,255,0.25)", width=1),
        name="EMA", showlegend=False
    ), row=1, col=1)

    # ─ Panel 2: SqueezeIndex ───────────────────────────────────────────────
    si_vals = df["SqueezeIndex"].values
    si_colors = [
        COLORS["red"] if v > 80 else COLORS["yellow"] if v > 50 else COLORS["blue"]
        for v in si_vals
    ]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["SqueezeIndex"],
        marker=dict(color=si_colors, opacity=0.9),
        name="SqueezeIndex",
    ), row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["red"], line_width=0.8,
                  annotation_text="Extremo", annotation_font=dict(size=9, color=COLORS["red"]),
                  row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["yellow"], line_width=0.8,
                  annotation_text="Moderado", annotation_font=dict(size=9, color=COLORS["yellow"]),
                  row=2, col=1)

    # ─ Panel 3: Trend ──────────────────────────────────────────────────────
    trend_vals = df["Trend"].values
    trend_colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in trend_vals]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Trend"],
        marker=dict(color=trend_colors, opacity=0.85),
        name="Trend",
    ), row=3, col=1)
    fig.add_hline(y=threshold, line_dash="dot", line_color="rgba(0,214,143,0.4)", line_width=0.8,
                  row=3, col=1)
    fig.add_hline(y=-threshold, line_dash="dot", line_color="rgba(255,71,87,0.4)", line_width=0.8,
                  row=3, col=1)
    fig.add_hline(y=0, line_color="#2a3440", line_width=1, row=3, col=1)

    # ─ Panel 4: Lambda ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Lambda"],
        line=dict(color=COLORS["purple"], width=1.5),
        fill="tozeroy", fillcolor="rgba(180,142,255,0.08)",
        name="Lambda Ω",
    ), row=4, col=1)

    # ─ Anotaciones de panel ───────────────────────────────────────────────
    annotations = [
        dict(text="PRECIO + BB + KELTNER", xref="paper", yref="paper",
             x=0.01, y=1.0, xanchor="left", showarrow=False,
             font=dict(size=9, color=COLORS["text_dim"], family="IBM Plex Mono")),
        dict(text="SQUEEZE INDEX (energía acumulada 0–100)", xref="paper", yref="paper",
             x=0.01, y=0.505, xanchor="left", showarrow=False,
             font=dict(size=9, color=COLORS["text_dim"], family="IBM Plex Mono")),
        dict(text="TREND COMPUESTO (4 componentes)", xref="paper", yref="paper",
             x=0.01, y=0.315, xanchor="left", showarrow=False,
             font=dict(size=9, color=COLORS["text_dim"], family="IBM Plex Mono")),
        dict(text="LAMBDA Ω — CICLO DOMINANTE (días)", xref="paper", yref="paper",
             x=0.01, y=0.145, xanchor="left", showarrow=False,
             font=dict(size=9, color=COLORS["text_dim"], family="IBM Plex Mono")),
    ]

    fig.update_layout(
        height=1000,
        margin=dict(l=8, r=8, t=8, b=8),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0d1117",
            bordercolor="#1e2630",
            font=dict(family="IBM Plex Mono, monospace", size=11)
        ),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="IBM Plex Mono, monospace", size=10, color=COLORS["text"]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            itemsizing="constant"
        ),
        xaxis_rangeslider_visible=False,
        annotations=annotations,
    )

    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor="#151c24", showgrid=True, zeroline=False,
            tickfont=dict(size=9), row=i, col=1
        )
        fig.update_yaxes(
            gridcolor="#151c24", showgrid=True, zeroline=False,
            tickfont=dict(size=9), row=i, col=1
        )

    return fig




def build_scan_chart(df_res: pd.DataFrame) -> go.Figure:
    df_sorted = df_res.sort_values("SI")
    colors = [COLORS["green"] if "🚨" in s else COLORS["yellow"] if "⏳" in s else COLORS["text_dim"]
              for s in df_sorted["Señal"].tolist()]
    fig = go.Figure(go.Bar(
        x=df_sorted["SI"],
        y=df_sorted["Activo"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v:.1f}" for v in df_sorted["SI"]],
        textposition="inside",
        textfont=dict(size=11, family="IBM Plex Mono, monospace", color="#fff"),
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="IBM Plex Mono, monospace", size=10, color=COLORS["text"]),
        xaxis=dict(title="SqueezeIndex", gridcolor="#151c24", range=[0, 105]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    fig.add_vline(x=80, line_dash="dot", line_color=COLORS["red"], line_width=1,
                  annotation_text="Extremo", annotation_font=dict(size=9, color=COLORS["red"]))
    fig.add_vline(x=50, line_dash="dot", line_color=COLORS["yellow"], line_width=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, sub: str = "", color: str = "blue") -> str:
    return f"""
    <div class="kpi-card {color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value kpi-{color}">{value}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def render_kpis(kpis: list):
    """Render a row of KPI cards. kpis = [(label, value, sub, color), ...]"""
    cols = st.columns(len(kpis))
    for col, (label, value, sub, color) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(label, value, sub, color), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════════════════════

def send_email_resend(to_email: str, subject: str, body: str):
    RESEND_KEY = st.secrets["RESEND_API_KEY"]
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_KEY}", "Content-Type": "application/json"},
            json={"from": "SqueezeIndex v3 <onboarding@resend.dev>",
                  "to": [to_email], "subject": subject, "text": body},
            timeout=15
        )
        return (True, "OK") if r.status_code == 200 else (False, r.text[:150])
    except Exception as e:
        return False, str(e)[:120]


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════════════

tab_dash, tab_scan, tab_metodologia = st.tabs([
    "Dashboard", "Escaneo Multi-Activo", "Metodología"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dash:
    if update:
        with st.spinner(f"Descargando datos de {selected_asset}…"):
            df_raw = fetch_data(ticker, days)

        if df_raw.empty:
            st.error("No se obtuvieron datos. Verifica el activo o la API key.")
        else:
            with st.spinner("Calculando modelo…"):
                df = calculate_squeeze_index(
                    df_raw.copy(), window, bb_mult, kc_mult,
                    atr_period, threshold, use_spectrum, use_vol_filter
                )

            if len(df) < 30:
                st.warning("Datos insuficientes para el cálculo (mínimo 30 barras).")
            else:
                last = df.iloc[-1]
                sq_days = int(df["SqueezeOn"].sum())
                num_ep = int(df["SqueezeEpisode"].max())
                pct_sq = sq_days / len(df) * 100
                avg_si = df.loc[df["SqueezeOn"], "SqueezeIndex"].mean() if sq_days > 0 else 0

                # ── Banner de estado ────────────────────────────────────────
                if last["SqueezeDetected"]:
                    dir_icon = "↑" if last["Direction"] == "Alcista" else "↓"
                    st.markdown(f"""
                    <div class="signal-banner signal-active">
                        🚨 SEÑAL ACTIVA &nbsp;·&nbsp; {dir_icon} {last['Direction'].upper()} &nbsp;·&nbsp;
                        Fuerza {last['SignalStrength']:.0%} &nbsp;·&nbsp;
                        SqueezeIndex {last['SqueezeIndex']:.1f}/100
                    </div>""", unsafe_allow_html=True)
                elif last["SqueezeOn"]:
                    st.markdown(f"""
                    <div class="signal-banner signal-pending">
                        ⏳ EN COMPRESIÓN — Acumulando energía &nbsp;·&nbsp;
                        Trend actual: {last['Trend']:.3f} (umbral: {threshold}) &nbsp;·&nbsp;
                        Sin señal todavía
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="signal-banner signal-none">
                        ● Sin compresión activa — El precio se mueve con amplitud normal
                    </div>""", unsafe_allow_html=True)

                # ── KPIs actuales ────────────────────────────────────────────
                st.markdown('<div class="section-header">SITUACIÓN ACTUAL</div>', unsafe_allow_html=True)

                si_color = "red" if last["SqueezeIndex"] > 80 else "yellow" if last["SqueezeIndex"] > 50 else "blue"
                trend_color = "green" if last["Trend"] > threshold else "red" if last["Trend"] < -threshold else "yellow"
                dir_color = "green" if last["Direction"] == "Alcista" else "red" if last["Direction"] == "Bajista" else "yellow"

                render_kpis([
                    ("Squeeze Index", f"{last['SqueezeIndex']:.1f}",
                     "0=sin tensión · 100=máxima compresión", si_color),
                    ("Trend", f"{last['Trend']:+.3f}",
                     f"Umbral señal: ±{threshold:.2f}", trend_color),
                    ("Dirección", last["Direction"],
                     "Hacia dónde apunta el modelo", dir_color),
                    ("Lambda Ω", f"{last['Lambda']:.1f}d",
                     "Ciclo dominante del activo", "purple"),
                    ("ATR", f"{last['ATR']:.4f}",
                     "Volatilidad real diaria", "blue"),
                ])

                st.markdown("")
                # ── KPIs del período ─────────────────────────────────────────
                st.markdown('<div class="section-header">ESTADÍSTICAS DEL PERÍODO</div>', unsafe_allow_html=True)
                render_kpis([
                    ("Días en compresión", f"{sq_days}",
                     f"{pct_sq:.1f}% del período analizado", "blue"),
                    ("Episodios detectados", f"{num_ep}",
                     "Squeezes de ≥3 días consecutivos", "purple"),
                    ("SI promedio (en squeeze)", f"{avg_si:.1f}",
                     "Intensidad media cuando hay compresión", "yellow"),
                    ("Barras totales", f"{len(df)}",
                     f"Desde {df['Date'].iloc[0]} hasta {df['Date'].iloc[-1]}", "blue"),
                    ("BB Width actual", f"{last['BBWidth']:.4f}",
                     "Anchura relativa de las bandas", "blue"),
                ])

                st.markdown("")
                st.markdown('<div class="section-header">GRÁFICO PRINCIPAL</div>', unsafe_allow_html=True)

                # Leyenda inline
                leg_col1, leg_col2, leg_col3 = st.columns(3)
                with leg_col1:
                    st.markdown("""
                    <div class="explain-box">
                    <b style="color:#4da6ff">Bandas de Bollinger (azul)</b> — Rango estadístico normal del precio.<br>
                    Cuando se estrechan, el precio está <em>comprimido</em>.
                    </div>""", unsafe_allow_html=True)
                with leg_col2:
                    st.markdown("""
                    <div class="explain-box">
                    <b style="color:#ffc107">Canal de Keltner (amarillo)</b> — Rango basado en volatilidad real (ATR).<br>
                    El <em>squeeze</em> ocurre cuando BB entra <em>dentro</em> de KC.
                    </div>""", unsafe_allow_html=True)
                with leg_col3:
                    st.markdown("""
                    <div class="explain-box">
                    <b style="color:#00d68f">▲ Señales (triángulos)</b> — Momento en que hay compresión + dirección clara.<br>
                    Verde = alcista · Rojo = bajista
                    </div>""", unsafe_allow_html=True)

                fig = build_main_chart(df, selected_asset)
                st.plotly_chart(fig, use_container_width=True)

                # Panel de información de indicadores
                with st.expander("📖 ¿Qué significa cada panel del gráfico?", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("""
                        **Panel 1 — Precio + Bandas**
                        - El fondo verde indica que hay squeeze activo (compresión)
                        - BB (azul): rango estadístico. KC (amarillo): rango de volatilidad
                        - Los triángulos marcan señales fuertes con dirección

                        **Panel 2 — SqueezeIndex (0 a 100)**
                        - Azul (<50): compresión baja, mercado normal
                        - Amarillo (50–80): compresión moderada, atención
                        - Rojo (>80): compresión extrema, alta probabilidad de ruptura inminente
                        """)
                    with c2:
                        st.markdown("""
                        **Panel 3 — Trend Compuesto**
                        - Verde: presión compradora dominante (señal alcista)
                        - Rojo: presión vendedora dominante (señal bajista)
                        - Las líneas punteadas marcan el umbral mínimo para activar señal
                        - Calculado con 4 componentes: slope largo, slope corto, MFI y ROC

                        **Panel 4 — Lambda Ω (ciclo dominante)**
                        - Días que dura cada ciclo de precio en este activo
                        - Estimado con análisis espectral Welch (muy robusto al ruido)
                        - Se usa para calibrar el Trend y el SqueezeIndex automáticamente
                        """)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 80px 20px; color: #6b7685;">
            <div style="font-size:48px; margin-bottom:16px;">〰️</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:16px; margin-bottom:8px;">
                Selecciona un activo y pulsa <b style="color:#4da6ff">▶ Calcular</b>
            </div>
            <div style="font-size:13px;">en la barra lateral para iniciar el análisis</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: ESCANEO MULTI-ACTIVO
# ─────────────────────────────────────────────────────────────────────────────
with tab_scan:
    st.markdown("## 🔭 Escaneo Multi-Activo")
    st.caption("Analiza todos los activos y muestra cuáles tienen señales activas ahora.")

    scan_days = st.slider("Días de histórico para el escaneo", 60, 365, 120, key="scan_days")
    scan_btn = st.button("🔭 Escanear todos los activos", type="primary")

    if scan_btn:
        results = []
        prog_bar = st.progress(0)
        status_txt = st.empty()
        asset_list = list(ASSETS.items())

        for idx, (name, tk) in enumerate(asset_list):
            prog_bar.progress((idx + 1) / len(asset_list))
            status_txt.markdown(f"<small style='color:#6b7685'>Analizando {name}…</small>", unsafe_allow_html=True)
            df_s = fetch_data(tk, scan_days)
            if df_s.empty or len(df_s) < 40:
                continue
            df_s = calculate_squeeze_index(
                df_s, window, bb_mult, kc_mult, atr_period, threshold, use_spectrum, use_vol_filter
            )
            if len(df_s) < 10:
                continue
            last_s = df_s.iloc[-1]

            results.append({
                "Activo": name,
                "Señal": "🚨 FUERTE" if last_s["SqueezeDetected"] else (
                         "⏳ LEVE" if last_s["SqueezeOn"] else "—"),
                "SI": round(last_s["SqueezeIndex"], 1),
                "Trend": round(last_s["Trend"], 3),
                "Dirección": last_s["Direction"],
                "Lambda Ω": round(last_s["Lambda"], 1),
                "Precio": round(last_s["Close"], 4),
                "Ep. históricos": int(df_s["SqueezeEpisode"].max()),
            })

        prog_bar.empty()
        status_txt.empty()

        if results:
            df_res = pd.DataFrame(results).sort_values("SI", ascending=False)

            # Gráfico de barras SI
            st.markdown('<div class="section-header">SQUEEZE INDEX ACTUAL POR ACTIVO</div>',
                        unsafe_allow_html=True)
            fig_scan = build_scan_chart(df_res)
            st.plotly_chart(fig_scan, use_container_width=True)

            # Tabla
            st.markdown('<div class="section-header">TABLA COMPLETA</div>', unsafe_allow_html=True)
            st.dataframe(
                df_res.style.apply(
                    lambda col: [
                        "background-color: rgba(0,214,143,0.1); color: #00d68f; font-weight:600"
                        if v == "🚨 FUERTE" else
                        "background-color: rgba(255,193,7,0.08); color: #ffc107"
                        if v == "⏳ LEVE" else ""
                        for v in col
                    ] if col.name == "Señal" else [""] * len(col),
                    axis=0
                ),
                use_container_width=True,
                height=380,
            )

            # Email
            st.divider()
            ecol1, ecol2 = st.columns([3, 1])
            with ecol1:
                user_email = st.text_input("📧 Email para recibir el reporte:", placeholder="tu@email.com")
            with ecol2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                send_btn = st.button("Enviar reporte")

            if send_btn and user_email:
                active = df_res[df_res["Señal"] != "—"]
                signals_str = "\n".join(
                    f"• {r['Activo']}: {r['Señal']} | {r['Dirección']} | SI {r['SI']} "
                    for _, r in active.iterrows()
                ) or "Sin señales activas en este momento."
                body = (
                    f"SqueezeIndex v3.0 — Escaneo {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"Parámetros: window={window} · BB={bb_mult} · KC={kc_mult} · threshold={threshold}\n\n"
                    f"=== SEÑALES ACTIVAS ===\n{signals_str}\n\n"
                    f"=== TODOS LOS ACTIVOS ===\n"
                    + "\n".join(f"• {r['Activo']}: SI={r['SI']} | {r['Dirección']}" for _, r in df_res.iterrows())
                )
                ok, msg = send_email_resend(user_email, f"Squeeze Scan {datetime.now(UTC).date()}", body)
                st.success("Enviado ✅") if ok else st.error(f"Error: {msg}")
        else:
            st.warning("No se obtuvieron resultados. Revisa la API key.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB: METODOLOGÍA
# ─────────────────────────────────────────────────────────────────────────────
with tab_metodologia:
    st.markdown("## 🧠 Metodología — SqueezeIndex v3.0")

    st.markdown("""
    <div class="explain-box">
    El modelo trata el precio como una <b>señal ondulatoria</b>. Como el mar: hay momentos de calma
    (compresión, energía acumulada) y momentos de oleaje (expansión, energía liberada).
    La hipótesis central es que la magnitud de la expansión es proporcional a la energía acumulada durante la compresión.
    </div>
    """, unsafe_allow_html=True)

    tab_glosario, tab_calculo, tab_mejoras, tab_limitaciones = st.tabs([
        "📖 Glosario", "🔢 Cómo se calcula", "⬆️ Mejoras v3.0", "⚠️ Limitaciones"
    ])

    with tab_glosario:
        st.markdown("""
        ### Conceptos clave (de más simple a más técnico)

        **Squeeze (compresión)**
        El precio lleva varios días moviéndose poco, como si estuviera "apretado" entre dos muros.
        En términos técnicos: las Bandas de Bollinger (basadas en desviación estándar) entran dentro
        del Canal de Keltner (basado en ATR). Mínimo de duración: **3 días consecutivos** (un episodio).

        **SqueezeIndex (0 a 100)**
        Cuánta tensión hay acumulada en este momento. Se calcula combinando:
        - Percentil del ancho de BB respecto a su propio historial (qué tan estrecho está *para este activo*)
        - Factor de calidad Lambda (Lorentziana) — penaliza si el ciclo es irregular

        Un valor de 80+ no significa "comprar/vender". Significa que la probabilidad de un movimiento
        grande *en algún momento próximo* es estadísticamente más alta que la media.

        **Lambda Ω (ciclo dominante)**
        Cada activo tiene su propio ritmo. El oro oscila más despacio que el Bitcoin.
        Lambda mide cuántos días dura un ciclo completo (de valle a pico a valle) usando
        análisis espectral de Welch, que es mucho más robusto que simplemente contar picos.

        **Trend (dirección de la tensión)**
        Cuando hay compresión, el modelo intenta determinar hacia dónde *ya está inclinado* el precio
        mediante 4 componentes ponderados: pendiente larga, pendiente corta, flujo de dinero (MFI) y velocidad (ROC).

        **Señal SqueezeDetected**
        La señal más restrictiva. Se activa solo cuando:
        1. Hay squeeze activo (SqueezeOn = ✓)
        2. El Trend supera el umbral configurado
        3. El ATR no está en el 25% más alto de su propio historial (mercado ya no está explotando)

        **Episodio**
        Período mínimo de 3 días consecutivos de compresión. El backtest opera al *final* de cada episodio.
        """)

    with tab_calculo:
        st.markdown("""
        ### Fórmulas detalladas

        **Bandas de Bollinger:**
        ```
        EMA(n) = media exponencial de cierre en n períodos
        STD(n) = desviación estándar de cierre en n períodos
        Upper BB = EMA + mult_BB × STD
        Lower BB = EMA - mult_BB × STD
        BB Width = (Upper BB - Lower BB) / EMA
        ```

        **Canal de Keltner:**
        ```
        ATR(n) = media exponencial del True Range en n períodos
        Upper KC = EMA + mult_KC × ATR
        Lower KC = EMA - mult_KC × ATR
        ```

        **SqueezeOn:**
        ```
        SqueezeOn = (Upper BB ≤ Upper KC) AND (Lower BB ≥ Lower KC)
        ```

        **Lambda Ω (Welch PSD):**
        ```
        1. Tomar ventana de N precios suavizados (EMA5)
        2. Calcular densidad espectral de potencia con método Welch
        3. Encontrar frecuencia dominante f*
        4. Lambda = 1 / f*  (longitud de onda en días)
        ```

        **SqueezeIndex:**
        ```
        bb_percentile = rank_pct(BBWidth, ventana=3×N)  # percentil histórico
        compression = 1 - bb_percentile  # 0=normal, 1=máxima compresión
        lambda_quality = 1 / (1 + ((Lambda - N/3) / (N/4))²)  # Lorentziana
        raw_SI = compression / BBWidth
        SI = (raw_SI / max_rolling(raw_SI)) × 100 × lambda_quality
        ```

        **Trend (4 componentes):**
        ```
        slope_largo  = pendiente(precios, ventana=Lambda) / ATR  → tanh(×5)  [35%]
        slope_corto  = pendiente(precios, ventana=6) / ATR        → tanh(×5)  [25%]
        MFI_score    = (MFI - 50) / 50                            [25%]
        ROC_norm     = tanh(ROC_5d / (ATR/Close) × 3)            [15%]
        Trend = EMA2(0.35×SL + 0.25×SC + 0.25×MFI + 0.15×ROC)
        ```
        """)

    with tab_mejoras:
        st.markdown("""
        | Componente | v2.1 | v3.0 | Impacto |
        |---|---|---|---|
        | **KPIs Dashboard** | Métricas en bruto sin contexto | KPI cards con color + explicación inline | Más accionable |
        | **Backtest** | Retorno raw sin alineación de dirección | Retorno alineado (long/short) + MFE/MAE + equity curve | Mide poder predictivo real |
        | **Métricas backtest** | Solo win rate y retorno medio | Win rate, expectancy, profit factor, Sharpe, drawdown, calmar | Visión completa del edge |
        | **Gráficos** | Colores inconsistentes, sin leyendas | Paleta unificada, leyendas inline, anotaciones de panel | Más legible |
        | **Indicadores** | Sin umbral visual en Trend | Líneas punteadas en ±threshold | Saber cuándo se activa la señal |
        | **Escaneo** | Tabla sola | Tabla + gráfico de barras horizontal | Más rápido de leer |
        | **Retorno en backtest** | Solo "retorno de precio" | Retorno alineado: alcista=long, bajista=short | Mide el modelo, no el mercado |
        """)

    with tab_limitaciones:
        st.markdown("""
        ### Lo que este modelo NO hace

        ⚠️ **No predice el momento exacto de la ruptura**. Solo indica que existe energía acumulada.
        La ruptura puede ocurrir en 1 día o en 15 días.

        ⚠️ **No incluye costes de transacción**. Los retornos del backtest son brutos.
        En mercados reales, spreads, comisiones y slippage reducirán los números.

        ⚠️ **Sample size limitado**. Con 365 días y parámetros por defecto, pueden generarse
        solo 8–15 episodios. Eso es insuficiente para conclusiones estadísticas sólidas.
        Usa 2–3 años mínimo para el backtest.

        ⚠️ **Sin gestión de posición**. El backtest asume entrada/salida fija.
        Un sistema real necesitaría stops, targets y sizing.

        ⚠️ **Look-ahead bias eliminado** (la señal se genera *antes* de medir el retorno),
        pero el backtest sigue siendo in-sample. Para validación real, reservar 1/3 de los datos
        como out-of-sample.

        ### Uso recomendado

        ✅ Como **scanner de atención**: qué activos tienen energía acumulada ahora mismo.

        ✅ Como **contexto de decisión**: el squeeze no es entrada, es contexto para otras señales.

        ✅ Como **análisis de régimen**: detectar si el mercado está en compresión o expansión.
        """)

    st.success("✅ SqueezeIndex v3.0 — Diseño claro · Backtest riguroso · Metodología transparente")
