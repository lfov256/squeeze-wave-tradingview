import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import linregress
from datetime import datetime, timedelta, UTC
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.set_page_config(
    page_title="SqueezeIndex v3",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;800&display=swap');
  html, body, [class*="css"] { font-family: 'Calibri', monospace; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
  .metric-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d; border-radius: 8px; padding: 16px;
    font-family: 'Space Mono', monospace;
  }
  .signal-strong { color: #00ff88; font-weight: 700; }
  .signal-mild { color: #ffcc00; font-weight: 700; }
  .signal-none { color: #6e7681; }
</style>
""", unsafe_allow_html=True)
st.title("〰️ SqueezeIndex v3.0")
st.caption(f"Ejecución: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
# ── Pestañas ───────────────────────────────────────────────────────────────────
tab_dash, tab_backtest, tab_scan, tab_info = st.tabs([
    "📈 Dashboard", "🎯 Backtest Real", "🔭 Escaneo Multi-Activo", "🧠 Metodología"
])
# ══════════════════════════════════════════════════════════════════════════════
# BARRA LATERAL
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🎛️ Control de Ondas")
    ASSETS = {
        "EURUSD": "C:EURUSD", "GBPUSD": "C:GBPUSD",
        "USDJPY": "C:USDJPY", "Oro": "C:XAUUSD",
        "Plata": "C:XAGUSD", "SPY": "SPY",
        "BTCUSD": "X:BTCUSD", "ETHUSD": "X:ETHUSD",
        "USO": "USO", "QQQ": "QQQ",
    }
    selected_asset = st.selectbox("Activo principal", list(ASSETS.keys()))
    ticker = ASSETS[selected_asset]
    days = st.slider("Días de histórico", 60, 730, 365)
    st.subheader("⚙️ Parámetros")
    window = st.slider("Ventana BB / EMA", 10, 60, 20)
    bb_mult = st.slider("BB Multiplier", 1.0, 3.5, 2.0, 0.1)
    kc_mult = st.slider("Keltner Multiplier", 1.0, 3.0, 1.5, 0.1)
    atr_period = st.slider("ATR Period", 10, 40, 20)
    threshold = st.slider("Trend Threshold", 0.05, 0.5, 0.15, 0.01)
    st.subheader("🔬 Análisis espectral")
    use_spectrum = st.checkbox("Usar frecuencia dominante (Welch)", value=True)
    update = st.button("🔄 Calcular", type="primary", use_container_width=True)
# ══════════════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════════════
API_KEY = st.secrets["API_KEY"]
# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES CORE — OPTIMIZADAS v3.0
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def fetch_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """Descarga OHLCV desde Polygon.io (sin cambios)."""
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
def compute_lambda_vectorized(smoothed: pd.Series, window: int, use_spectrum: bool) -> pd.Series:
    """
    v3.0 — Welch PSD robusto + vectorización parcial.
    nperseg más estable (mínimo 16, máximo razonable) → frecuencia dominante fiable.
    """
    n = len(smoothed)
    lambda_arr = np.full(n, np.nan)
    prices_arr = smoothed.values
    for i in range(window - 1, n):
        seg = prices_arr[i - window + 1 : i + 1]
        seg_len = len(seg)
       
        if use_spectrum and seg_len >= 16:
            try:
                # OPTIMIZACIÓN v3: nperseg dinámico pero estable (evita ventanas de 8 puntos)
                nperseg = max(16, min(seg_len // 2, 64))
                freqs, psd = welch(seg, nperseg=nperseg, scaling='density')
                freqs = freqs[freqs > 0]
                psd = psd[1:] if len(psd) > 1 else psd
                if len(psd) > 0:
                    dom_freq = freqs[np.argmax(psd)]
                    lambda_arr[i] = 1.0 / dom_freq if dom_freq > 1e-8 else window / 2.0
                else:
                    lambda_arr[i] = window / 2.0
            except Exception:
                lambda_arr[i] = window / 2.0
        else:
            # Peaks fallback (rápido)
            peaks, _ = find_peaks(seg)
            valleys, _ = find_peaks(-seg)
            extrema = np.sort(np.concatenate([peaks, valleys]))
            if len(extrema) > 1:
                lambda_arr[i] = float(np.mean(np.diff(extrema)))
            else:
                lambda_arr[i] = window / 2.0
    lam = pd.Series(lambda_arr, index=smoothed.index)
    lam = lam.ffill().bfill().clip(lower=2.0, upper=window * 2)
    return lam
def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR vectorizado (sin cambios)."""
    prev_c = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"] - prev_c).abs()
    ], axis=1).max(axis=1).fillna(0)
    return tr.ewm(span=period, adjust=False).mean()
def compute_energy_accumulated(df: pd.DataFrame, window: int) -> pd.Series:
    """
    v3.0 — FÍSICA PURA: Energía acumulada durante compresión.
    E_t = Σ [1 / (ATR_i / ATR_media)] durante SqueezeOn (integral inversa de volatilidad).
    Representa la "carga" del oscilador subamortiguado.
    """
    atr_mean = df["ATR"].rolling(window * 3, min_periods=10).mean()
    squeeze_mask = df["SqueezeOn"].astype(float)
   
    # Contribución inversa normalizada (cuanto más baja la ATR durante squeeze → más energía)
    contrib = squeeze_mask / (df["ATR"] / atr_mean.replace(0, np.nan)).clip(lower=0.1)
    # Rolling sum reciente (energía "cargada" en las últimas ~2 ventanas)
    energy = contrib.fillna(0).rolling(window * 2, min_periods=5).sum()
    # Normalización 0-1 relativa al histórico reciente
    energy_norm = (energy / energy.rolling(window * 3).max().replace(0, np.nan)).fillna(0).clip(0, 1)
    return energy_norm
def compute_trend(df: pd.DataFrame, smoothed: pd.Series, lam: pd.Series) -> pd.Series:
    """Trend compuesto 4 componentes — loop optimizado (rápido para datos diarios)."""
    n = len(df)
    prices_arr = smoothed.values
    atr_arr = df["ATR"].values
    lam_arr = lam.values
    slope_long_arr = np.zeros(n)
    slope_short_arr = np.zeros(n)
    for i in range(n):
        lv = max(6, int(lam_arr[i]))
        atr_v = atr_arr[i]
        if atr_v <= 0 or np.isnan(atr_v):
            continue
        # Long slope (ventana dinámica = Lambda)
        if i >= lv - 1:
            seg = prices_arr[i - lv + 1 : i + 1]
            x = np.arange(len(seg), dtype=float)
            slope_long_arr[i] = linregress(x, seg).slope / atr_v # linregress más rápido que polyfit(1)
        # Short slope (fija 6 barras)
        short_w = 6
        if i >= short_w - 1:
            seg_s = prices_arr[i - short_w + 1 : i + 1]
            x_s = np.arange(short_w, dtype=float)
            slope_short_arr[i] = linregress(x_s, seg_s).slope / atr_v
    slope_long_norm = np.tanh(slope_long_arr * 5)
    slope_short_norm = np.tanh(slope_short_arr * 5)
    # MFI + ROC (sin cambios, ya vectorizados)
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
    """Etiqueta episodios (sin cambios)."""
    episode = np.zeros(len(squeeze_on), dtype=int)
    ep_id = 0
    in_sq = False
    start = 0
    for i, val in enumerate(squeeze_on):
        if val and not in_sq:
            in_sq = True
            start = i
        elif not val and in_sq:
            in_sq = False
            duration = i - start
            if duration >= min_duration:
                ep_id += 1
                episode[start:i] = ep_id
        elif val and in_sq and i == len(squeeze_on) - 1:
            duration = i - start + 1
            if duration >= min_duration:
                ep_id += 1
                episode[start:] = ep_id
    return pd.Series(episode, index=squeeze_on.index)
def calculate_squeeze_index(
    df: pd.DataFrame,
    window: int, bb_mult: float, kc_mult: float,
    atr_period: int, threshold: float,
    use_spectrum: bool = True
) -> pd.DataFrame:
    """
    Pipeline v3.0: añade Energía Acumulada explícita (física del oscilador).
    """
    if df.empty or len(df) < window + 10:
        return df
    # ── Bandas de Bollinger + Keltner + ATR (igual) ───────────────────────
    df["EMA"] = df["Close"].ewm(span=window, adjust=False).mean()
    df["STD"] = df["Close"].rolling(window).std()
    df["UpperBB"] = df["EMA"] + bb_mult * df["STD"]
    df["LowerBB"] = df["EMA"] - bb_mult * df["STD"]
    df["BBWidth"] = (df["UpperBB"] - df["LowerBB"]) / df["EMA"].replace(0, np.nan)
    df["ATR"] = compute_atr(df, atr_period)
    df["UpperKC"] = df["EMA"] + kc_mult * df["ATR"]
    df["LowerKC"] = df["EMA"] - kc_mult * df["ATR"]
    # ── Lambda Ω (Welch optimizado) ────────────────────────────────────────
    smoothed = df["Close"].ewm(span=5, adjust=False).mean()
    df["Lambda"] = compute_lambda_vectorized(smoothed, window, use_spectrum)
    # ── SqueezeOn + Episodios (igual) ──────────────────────────────────────
    df["SqueezeOn"] = (df["UpperBB"] <= df["UpperKC"]) & (df["LowerBB"] >= df["LowerKC"])
    df["SqueezeEpisode"] = detect_squeeze_episodes(df["SqueezeOn"], min_duration=3)
    # ── ENERGÍA ACUMULADA v3.0 (nuevo) ─────────────────────────────────────
    df["Energy"] = compute_energy_accumulated(df, window)
    # ── SqueezeIndex v3: compresión + lambda_quality + ENERGY ──────────────
    bb_percentile = df["BBWidth"].rolling(window * 3).rank(pct=True).fillna(0.5)
    compression_factor = (1 - bb_percentile).clip(0, 1)
    lambda_quality = np.exp(-0.5 * ((df["Lambda"] - window / 3) / (window / 4)) ** 2)
   
    raw_si = compression_factor / df["BBWidth"].replace(0, np.nan)
    si_roll_max = raw_si.rolling(window * 3).max().replace(0, np.nan)
   
    # Incorporamos energía: cuanto más energía cargada → mayor índice
    df["SqueezeIndex"] = (
        (raw_si / si_roll_max) * 100 * lambda_quality * (0.7 + 0.3 * df["Energy"])
    ).clip(0, 100).fillna(0)
    # ── Trend + Dirección (igual) ──────────────────────────────────────────
    df["Trend"] = compute_trend(df, smoothed, df["Lambda"])
    df["Direction"] = np.where(df["Trend"] > 0, "Alcista",
                       np.where(df["Trend"] < 0, "Bajista", "Neutral"))
    # ── SqueezeDetected + SignalStrength (mejorado con Energy) ─────────────
    df["SqueezeDetected"] = df["SqueezeOn"] & (df["Trend"].abs() > threshold)
    df["SignalStrength"] = (
        0.5 * df["SqueezeIndex"] / 100 +
        0.3 * df["Trend"].abs().clip(0, 1) +
        0.2 * df["Energy"] # nuevo peso a la energía física
    ).clip(0, 1)
    return df
# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST REAL (EPISODIOS) — v3.0 con filtro de energía alta
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """Backtest POST-episodio (igual) + columna Energy_Max para análisis."""
    records = []
    episodes = df[df["SqueezeEpisode"] > 0]["SqueezeEpisode"].unique()
    for ep_id in episodes:
        ep_mask = df["SqueezeEpisode"] == ep_id
        ep_data = df[ep_mask]
        ep_end_i = ep_data.index[-1]
        ep_end_pos = df.index.get_loc(ep_end_i)
        fwd_end_pos = min(ep_end_pos + forward_days, len(df) - 1)
        if fwd_end_pos <= ep_end_pos:
            continue
        entry_price = df["Close"].iloc[ep_end_pos]
        exit_price = df["Close"].iloc[fwd_end_pos]
        fwd_return = (exit_price - entry_price) / entry_price * 100
        last_day = ep_data.iloc[-1]
        pred_dir = last_day["Direction"]
        trend_val = last_day["Trend"]
        si_max = ep_data["SqueezeIndex"].max()
        energy_max = ep_data["Energy"].max()
        duration = len(ep_data)
        actual_dir = "Alcista" if fwd_return > 0 else "Bajista" if fwd_return < 0 else "Neutral"
        hit = (pred_dir == actual_dir) and (pred_dir != "Neutral")
        records.append({
            "Episodio": ep_id,
            "Fin_Episodio": df["Date"].iloc[ep_end_pos],
            "Duración_días": duration,
            "SI_Max": round(si_max, 1),
            "Energy_Max": round(energy_max, 3),
            "Trend_Señal": round(trend_val, 4),
            "Dirección_Pred": pred_dir,
            "Precio_Entrada": round(entry_price, 4),
            "Precio_Salida": round(exit_price, 4),
            "Retorno_%": round(fwd_return, 2),
            "Dirección_Real": actual_dir,
            "Acierto": "✅" if hit else "❌",
        })
    return pd.DataFrame(records)
# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS — Añadido subplot de Energía
# ══════════════════════════════════════════════════════════════════════════════
def build_main_chart(df: pd.DataFrame, asset_name: str) -> go.Figure:
    """Gráfico principal v3.0: 5 subplots (añadida Energía)."""
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.15, 0.15, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{asset_name} — Precio + BB + Keltner",
            "SqueezeIndex v3 (compresión + λ + Energía)",
            "Energía Acumulada (física del oscilador)",
            "Trend Compuesto",
            "Lambda Ω — Longitud de Onda Dominante",
        )
    )
    # Candlestick + bandas (igual)
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Precio"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["UpperBB"], line=dict(color="#4488ff", width=1.2), name="Upper BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LowerBB"], line=dict(color="#4488ff", width=1.2), fill="tonexty", fillcolor="rgba(68,136,255,0.06)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["UpperKC"], line=dict(color="#ffaa00", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LowerKC"], line=dict(color="#ffaa00", width=1, dash="dash")), row=1, col=1)
    # SqueezeOn background y marcadores (igual)
    for i in range(len(df) - 1):
        if df["SqueezeOn"].iloc[i]:
            fig.add_vrect(x0=df["Date"].iloc[i], x1=df["Date"].iloc[i + 1], fillcolor="rgba(0,255,136,0.12)", layer="below", line_width=0, row=1, col=1)
    sq_det = df[df["SqueezeDetected"]]
    fig.add_trace(go.Scatter(x=sq_det["Date"], y=sq_det["Low"] * 0.998, mode="markers", marker=dict(symbol="triangle-up", color="gold", size=10), name="SqueezeDetected ▲"), row=1, col=1)
    # SqueezeIndex
    si_colors = np.where(df["SqueezeIndex"] > 80, "#ff4466", np.where(df["SqueezeIndex"] > 50, "#ffaa00", "#4488ff"))
    fig.add_trace(go.Bar(x=df["Date"], y=df["SqueezeIndex"], marker_color=si_colors), row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#ff4466", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#ffaa00", row=2, col=1)
    # Nueva fila: Energía Acumulada
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Energy"], line=dict(color="#bb88ff", width=2), fill="tozeroy", fillcolor="rgba(187,136,255,0.25)"), row=3, col=1)
    fig.add_hline(y=0.7, line_dash="dot", line_color="#00ff88", annotation_text="Alta energía", row=3, col=1)
    # Trend
    trend_colors = np.where(df["Trend"] > 0, "#00ff88", "#ff4466")
    fig.add_trace(go.Bar(x=df["Date"], y=df["Trend"], marker_color=trend_colors), row=4, col=1)
    fig.add_hline(y=0, line_color="#666", row=4, col=1)
    # Lambda
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Lambda"], line=dict(color="#bb88ff", width=1.5), fill="tozeroy"), row=5, col=1)
    fig.update_layout(height=1100, template="plotly_dark", hovermode="x unified", showlegend=False,
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      font=dict(family="Space Mono, monospace", size=11),
                      title=dict(text=f"〰️ {asset_name} — Ondas de Compresión v3.0 (Energía + λ)", font=dict(size=16)))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig
# (Las funciones build_backtest_chart, send_email_resend y las pestañas permanecen idénticas a v2.0 — solo se actualiza el llamado a calculate_squeeze_index y los gráficos)
# ══════════════════════════════════════════════════════════════════════════════
# TAB DASHBOARD (actualizado con Energy en métricas)
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    if update:
        with st.spinner(f"Descargando {selected_asset}…"):
            df_raw = fetch_data(ticker, days)
        if df_raw.empty:
            st.error("No se pudieron obtener datos.")
        else:
            with st.spinner("Calculando modelo v3.0…"):
                df = calculate_squeeze_index(
                    df_raw.copy(), window, bb_mult, kc_mult,
                    atr_period, threshold, use_spectrum
                )
            if len(df) < 30:
                st.warning("Datos insuficientes.")
            else:
                last = df.iloc[-1]
                st.success(f"✅ {selected_asset} · {len(df)} barras · Última: {last['Date']}")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("SqueezeIndex", f"{last['SqueezeIndex']:.1f}/100")
                c2.metric("Energy", f"{last['Energy']:.2f}")
                c3.metric("Trend", f"{last['Trend']:.3f}")
                c4.metric("Dirección", last["Direction"])
                c5.metric("Estado", "🟢 SqueezeOn" if last["SqueezeOn"] else "⚪ Sin compresión")
                c6.metric("Lambda Ω", f"{last['Lambda']:.1f} días")
                
