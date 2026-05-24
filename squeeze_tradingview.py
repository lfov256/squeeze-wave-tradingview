"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SQUEEZE INDEX v2.1 — Best of Both (Main + v2)                             ║
║   Metodología clara + Edge Cuantitativo Real (Welch + Episodios + Backtest) ║
║   Inspirado en Jim Simons / Renaissance Technologies                         ║
╚══════════════════════════════════════════════════════════════════════════════╗
"""
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
    page_title="SqueezeIndex v2.1",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos (mejorados de Main) ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;800&display=swap');
  html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
  .metric-card { 
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d; border-radius: 8px; padding: 16px;
  }
  .signal-strong { color: #00ff88; font-weight: 700; }
  .signal-mild   { color: #ffcc00; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("〰️ SqueezeIndex v2.1")
st.caption(f"Ejecución: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')} · Física de Ondas + Backtest Real + Metodología Clara")

# ── Pestañas ───────────────────────────────────────────────────────────────────
tab_dash, tab_backtest, tab_scan, tab_metodologia = st.tabs([
    "📈 Dashboard", "🎯 Backtest Real", "🔭 Escaneo Multi-Activo", "🧠 Metodología"
])

# ══════════════════════════════════════════════════════════════════════════════
# BARRA LATERAL (mejorada)
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
    window = st.slider("Ventana BB / EMA / Lambda", 10, 60, 20)
    bb_mult = st.slider("BB Multiplier", 1.0, 3.5, 2.0, 0.1)
    kc_mult = st.slider("Keltner Multiplier", 1.0, 3.0, 1.5, 0.1)
    atr_period = st.slider("ATR Period", 10, 40, 20)
    threshold = st.slider("Trend Threshold", 0.05, 0.5, 0.15, 0.01)

    st.subheader("🔬 Análisis espectral")
    use_spectrum = st.checkbox("Usar frecuencia dominante (Welch)", value=True)
    use_vol_filter = st.checkbox("Filtrar régimen alta volatilidad", value=True)

    update = st.button("🔄 Calcular", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════════════
API_KEY = st.secrets["API_KEY"]

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES CORE (v2 completo + mejoras)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """Descarga OHLCV desde Polygon.io (más estable que massive.com)."""
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
    """Lambda con Welch PSD vectorizado (robusto) o peaks clásico."""
    n = len(smoothed)
    lambda_arr = np.full(n, np.nan)
    prices_arr = smoothed.values

    for i in range(window - 1, n):
        seg = prices_arr[i - window + 1 : i + 1]
        if use_spectrum and len(seg) >= 16:
            try:
                freqs, psd = welch(seg, nperseg=min(len(seg), 8))
                freqs = freqs[freqs > 0]
                psd = psd[1:]
                if len(psd) > 0:
                    dom_freq = freqs[np.argmax(psd)]
                    lambda_arr[i] = 1.0 / dom_freq if dom_freq > 0 else window / 2
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
    lam = lam.ffill().bfill().clip(lower=2.0)
    return lam


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_c = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"] - prev_c).abs()
    ], axis=1).max(axis=1).fillna(0)
    return tr.ewm(span=period, adjust=False).mean()


def compute_trend(df: pd.DataFrame, smoothed: pd.Series, lam: pd.Series) -> pd.Series:
    """Trend 4 componentes (mejorado de v2)."""
    n = len(df)
    prices_arr = smoothed.values
    atr_arr = df["ATR"].values
    lam_arr = lam.values

    slope_long_arr = np.zeros(n)
    slope_short_arr = np.zeros(n)

    for i in range(n):
        lv = max(6, int(lam_arr[i] * 1.0))
        atr_v = atr_arr[i]
        if atr_v == 0 or np.isnan(atr_v):
            continue
        if i >= lv - 1:
            seg = prices_arr[i - lv + 1 : i + 1]
            x = np.arange(len(seg), dtype=float)
            slope_long_arr[i] = np.polyfit(x, seg, 1)[0] / atr_v
        short_w = 6
        if i >= short_w - 1:
            seg_s = prices_arr[i - short_w + 1 : i + 1]
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
    use_spectrum: bool = True, use_vol_filter: bool = True
) -> pd.DataFrame:
    if df.empty or len(df) < window + 10:
        return df

    # Bandas Bollinger
    df["EMA"] = df["Close"].ewm(span=window, adjust=False).mean()
    df["STD"] = df["Close"].rolling(window).std()
    df["UpperBB"] = df["EMA"] + bb_mult * df["STD"]
    df["LowerBB"] = df["EMA"] - bb_mult * df["STD"]
    df["BBWidth"] = (df["UpperBB"] - df["LowerBB"]) / df["EMA"].replace(0, np.nan)

    # ATR + Keltner
    df["ATR"] = compute_atr(df, atr_period)
    df["UpperKC"] = df["EMA"] + kc_mult * df["ATR"]
    df["LowerKC"] = df["EMA"] - kc_mult * df["ATR"]

    # Lambda (Welch o peaks)
    smoothed = df["Close"].ewm(span=5, adjust=False).mean()
    df["Lambda"] = compute_lambda_vectorized(smoothed, window, use_spectrum)

    # SqueezeIndex mejorado (percentil + Lorentzian)
    bb_percentile = df["BBWidth"].rolling(window * 3).rank(pct=True).fillna(0.5)
    compression_factor = (1 - bb_percentile).clip(0, 1)
    # Lorentzian (mejor para colas pesadas)
    lambda_quality = 1 / (1 + ((df["Lambda"] - window / 3) / (window / 4)) ** 2)
    raw_si = compression_factor / df["BBWidth"].replace(0, np.nan)
    si_roll_max = raw_si.rolling(window * 3).max().replace(0, np.nan)
    df["SqueezeIndex"] = ((raw_si / si_roll_max) * 100 * lambda_quality).clip(0, 100).fillna(0)

    # Episodios
    df["SqueezeOn"] = (df["UpperBB"] <= df["UpperKC"]) & (df["LowerBB"] >= df["LowerKC"])
    df["SqueezeEpisode"] = detect_squeeze_episodes(df["SqueezeOn"], min_duration=3)

    # Trend 4 componentes
    df["Trend"] = compute_trend(df, smoothed, df["Lambda"])
    df["Direction"] = np.where(df["Trend"] > 0, "Alcista",
                       np.where(df["Trend"] < 0, "Bajista", "Neutral"))

    # Señal + Filtro régimen vol
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
# BACKTEST REAL (EPISODIOS) — MEJORADO v2.1
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Backtest riguroso por EPISODIO de squeeze (el único matemáticamente correcto).
    Mide el retorno REAL en los N días SIGUIENTES al final del episodio.
    """
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
        fwd_high = df["High"].iloc[ep_end_pos + 1 : fwd_end_pos + 1].max()
        fwd_low = df["Low"].iloc[ep_end_pos + 1 : fwd_end_pos + 1].min()

        last_day = ep_data.iloc[-1]
        pred_dir = last_day["Direction"]
        trend_val = last_day["Trend"]
        si_max = ep_data["SqueezeIndex"].max()
        duration = len(ep_data)

        actual_dir = "Alcista" if fwd_return > 0 else "Bajista" if fwd_return < 0 else "Neutral"
        hit = (pred_dir == actual_dir) and (pred_dir != "Neutral")

        records.append({
            "Episodio": ep_id,
            "Fin_Episodio": df["Date"].iloc[ep_end_pos],
            "Duración_días": duration,
            "SI_Max": round(si_max, 1),
            "Trend_Señal": round(trend_val, 4),
            "Dirección_Pred": pred_dir,
            "Precio_Entrada": round(entry_price, 4),
            "Precio_Salida": round(exit_price, 4),
            "Retorno_%": round(fwd_return, 2),
            "High_fwd": round(fwd_high, 4),
            "Low_fwd": round(fwd_low, 4),
            "Dirección_Real": actual_dir,
            "Acierto": "✅" if hit else "❌",
        })

    bt = pd.DataFrame(records)
    if bt.empty:
        return bt

    # Métricas adicionales v2.1
    bt["Expectancy"] = bt.apply(
        lambda row: row["Retorno_%"] if row["Acierto"] == "✅" else -abs(row["Retorno_%"]), axis=1
    )
    wins = bt[bt["Acierto"] == "✅"]["Retorno_%"]
    losses = bt[bt["Acierto"] == "❌"]["Retorno_%"]
    if len(wins) > 0 and len(losses) > 0:
        win_rate = len(wins) / len(bt)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        bt["Sharpe_aprox"] = (bt["Retorno_%"].mean() / bt["Retorno_%"].std()) * np.sqrt(252) if bt["Retorno_%"].std() > 0 else 0
    else:
        bt["Sharpe_aprox"] = 0

    return bt


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def build_main_chart(df: pd.DataFrame, asset_name: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{asset_name} — Precio + BB + Keltner",
            "SqueezeIndex v2.1 (energía acumulada, escala 0-100)",
            "Trend Compuesto (4 componentes + ROC)",
            "Lambda Ω — Longitud de Onda Dominante (Welch)",
        )
    )

    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Precio",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4466"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["UpperBB"],
        line=dict(color="#4488ff", width=1.2), name="Upper BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LowerBB"],
        line=dict(color="#4488ff", width=1.2), fill="tonexty",
        fillcolor="rgba(68,136,255,0.06)", name="Lower BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["UpperKC"],
        line=dict(color="#ffaa00", width=1, dash="dash"), name="Upper KC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["LowerKC"],
        line=dict(color="#ffaa00", width=1, dash="dash"), name="Lower KC"), row=1, col=1)

    # Fondo verde = SqueezeOn
    for i in range(len(df) - 1):
        if df["SqueezeOn"].iloc[i]:
            fig.add_vrect(
                x0=df["Date"].iloc[i], x1=df["Date"].iloc[i + 1],
                fillcolor="rgba(0,255,136,0.12)", layer="below", line_width=0,
                row=1, col=1
            )

    # Marcadores SqueezeDetected
    sq_det = df[df["SqueezeDetected"]]
    fig.add_trace(go.Scatter(
        x=sq_det["Date"], y=sq_det["Low"] * 0.998,
        mode="markers",
        marker=dict(symbol="triangle-up", color="gold", size=10),
        name="SqueezeDetected ▲"
    ), row=1, col=1)

    # SqueezeIndex
    si_colors = np.where(
        df["SqueezeIndex"] > 80, "#ff4466",
        np.where(df["SqueezeIndex"] > 50, "#ffaa00", "#4488ff")
    )
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["SqueezeIndex"],
        marker_color=si_colors, name="SqueezeIndex"
    ), row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#ff4466",
                  annotation_text="Extremo", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#ffaa00",
                  annotation_text="Medio", row=2, col=1)

    # Trend
    trend_colors = np.where(df["Trend"] > 0, "#00ff88", "#ff4466")
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Trend"],
        marker_color=trend_colors, name="Trend"
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="#666", line_width=0.8, row=3, col=1)

    # Lambda
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Lambda"],
        line=dict(color="#bb88ff", width=1.5), name="Lambda Ω",
        fill="tozeroy", fillcolor="rgba(187,136,255,0.1)"
    ), row=4, col=1)

    fig.update_layout(
        height=950,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Space Mono, monospace", size=11),
        title=dict(text=f"〰️ {asset_name} — Ondas de Compresión v2.1", font=dict(size=16))
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", showgrid=True)
    return fig


def build_backtest_chart(bt: pd.DataFrame) -> go.Figure:
    if bt.empty:
        return go.Figure()

    colors = np.where(bt["Acierto"] == "✅", "#00ff88", "#ff4466")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bt["Fin_Episodio"].astype(str),
        y=bt["Retorno_%"],
        marker_color=colors,
        text=bt["Dirección_Pred"],
        textposition="outside",
        name="Retorno Fwd"
    ))
    fig.update_layout(
        title="Retorno post-squeeze por episodio (verdes = acierto)",
        template="plotly_dark",
        height=380,
        xaxis_title="Fin del episodio",
        yaxis_title="Retorno % (N días)",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Space Mono, monospace", size=11),
    )
    fig.add_hline(y=0, line_color="#888", line_width=0.8)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL (mejorado)
# ══════════════════════════════════════════════════════════════════════════════

def send_email_resend(to_email: str, subject: str, body: str):
    RESEND_KEY = st.secrets["RESEND_API_KEY"]
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_KEY}",
                     "Content-Type": "application/json"},
            json={"from": "SqueezeIndex v2.1 <onboarding@resend.dev>",
                  "to": [to_email], "subject": subject, "text": body},
            timeout=15
        )
        return (True, "OK") if r.status_code == 200 else (False, r.text[:150])
    except Exception as e:
        return False, str(e)[:120]

        # ══════════════════════════════════════════════════════════════════════════════
# TAB DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    if update:
        with st.spinner(f"Descargando {selected_asset}…"):
            df_raw = fetch_data(ticker, days)

        if df_raw.empty:
            st.error("No se pudieron obtener datos.")
        else:
            with st.spinner("Calculando modelo v2.1…"):
                df = calculate_squeeze_index(
                    df_raw.copy(), window, bb_mult, kc_mult,
                    atr_period, threshold, use_spectrum, use_vol_filter
                )

            if len(df) < 30:
                st.warning("Datos insuficientes.")
            else:
                last = df.iloc[-1]
                st.success(f"✅ {selected_asset} · {len(df)} barras · Última: {last['Date']}")

                # Métricas principales
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("SqueezeIndex", f"{last['SqueezeIndex']:.1f}/100")
                c2.metric("Trend", f"{last['Trend']:.3f}")
                c3.metric("Dirección", last["Direction"])
                c4.metric("Estado", "🟢 SqueezeOn" if last["SqueezeOn"] else "⚪ Sin compresión")
                c5.metric("Lambda Ω", f"{last['Lambda']:.1f} días")

                # Estadísticas
                sq_days = df["SqueezeOn"].sum()
                sq_det = df["SqueezeDetected"].sum()
                num_ep = df["SqueezeEpisode"].max()
                pct_sq = sq_days / len(df) * 100
                avg_si = df.loc[df["SqueezeOn"], "SqueezeIndex"].mean() if sq_days > 0 else 0

                st.divider()
                st.subheader("📊 Estadísticas del período")
                ca, cb, cc, cd, ce = st.columns(5)
                ca.metric("Días en compresión", f"{sq_days}")
                cb.metric("% tiempo comprimido", f"{pct_sq:.1f}%")
                cc.metric("Episodios detectados", f"{num_ep}")
                cd.metric("Señales fuertes", f"{sq_det}")
                ce.metric("SI promedio (squeeze)", f"{avg_si:.1f}")

                # Gráfico principal
                fig = build_main_chart(df, selected_asset)
                st.plotly_chart(fig, use_container_width=True)

                # Señal actual
                if last["SqueezeDetected"]:
                    st.success(
                        f"🚨 **SEÑAL ACTIVA** — {last['Direction']} | "
                        f"Fuerza {last['SignalStrength']:.2f} | "
                        f"SqueezeIndex {last['SqueezeIndex']:.1f}"
                    )
                elif last["SqueezeOn"]:
                    st.info(
                        f"⏳ En compresión — Trend insuficiente ({last['Trend']:.3f}). "
                        f"Threshold = {threshold}"
                    )
    else:
        st.info("👈 Selecciona activo y parámetros, luego pulsa **Calcular**.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB BACKTEST REAL
# ══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.subheader("🎯 Backtest Real — Episodios de Compresión (v2.1)")
    st.markdown("""
    **Metodología correcta**: Se registra la dirección prevista al **finalizar** cada episodio de squeeze 
    y se mide el retorno real en los N días **siguientes**. Esto es lo que importa para operar.
    """)

    fwd_days = st.slider("Ventana de medición post-squeeze (días)", 2, 20, 5, key="bt_fwd")

    if update:
        if "df" in dir() and not df.empty and "SqueezeEpisode" in df.columns:
            with st.spinner("Calculando backtest…"):
                bt = run_backtest(df, forward_days=fwd_days)

            if bt.empty:
                st.warning("No hay episodios suficientes (mínimo 3 días consecutivos).")
            else:
                hits = (bt["Acierto"] == "✅").sum()
                total = len(bt)
                pct_hit = hits / total * 100 if total > 0 else 0
                avg_ret_hit = bt.loc[bt["Acierto"] == "✅", "Retorno_%"].mean()
                avg_ret_miss = bt.loc[bt["Acierto"] == "❌", "Retorno_%"].mean()
                avg_si_hit = bt.loc[bt["Acierto"] == "✅", "SI_Max"].mean()

                # KPIs
                bk1, bk2, bk3, bk4, bk5 = st.columns(5)
                bk1.metric("Precisión global", f"{pct_hit:.1f}%", f"{hits}/{total}")
                bk2.metric("Episodios totales", f"{total}")
                bk3.metric("Retorno medio ✅", f"{avg_ret_hit:.2f}%")
                bk4.metric("Retorno medio ❌", f"{avg_ret_miss:.2f}%")
                bk5.metric("SI medio aciertos", f"{avg_si_hit:.1f}")

                # Gráfico
                fig_bt = build_backtest_chart(bt)
                st.plotly_chart(fig_bt, use_container_width=True)

                # Tabla detallada
                with st.expander("📋 Tabla completa de episodios", expanded=True):
    st.dataframe(
        bt.style.applymap(
            lambda v: "color: #00ff88" if v == "✅" else "color: #ff4466" if v == "❌" else "",
            subset=["Acierto"]
        ),
        use_container_width=True
    )
                # Análisis por intensidad
                st.subheader("🔬 ¿La intensidad del squeeze mejora la precisión?")
                q33 = bt["SI_Max"].quantile(0.33)
                q66 = bt["SI_Max"].quantile(0.66)
                bt["Intensidad"] = pd.cut(
                    bt["SI_Max"],
                    bins=[-np.inf, q33, q66, np.inf],
                    labels=["Baja", "Media", "Alta"]
                )
                intensidad_summary = bt.groupby("Intensidad").apply(
                    lambda g: pd.Series({
                        "N": len(g),
                        "Precisión": f"{(g['Acierto']=='✅').sum()/len(g)*100:.1f}%",
                        "Ret_medio": f"{g['Retorno_%'].mean():.2f}%",
                        "SI_medio": f"{g['SI_Max'].mean():.1f}",
                    })
                ).reset_index()
                st.dataframe(intensidad_summary, use_container_width=True)
        else:
            st.info("Primero pulsa **Calcular** en la pestaña Dashboard.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB ESCANEO MULTI-ACTIVO
# ══════════════════════════════════════════════════════════════════════════════

with tab_scan:
    st.subheader("🔭 Escaneo Multi-Activo")
    st.caption("Escanea todos los activos y muestra señales activas ahora mismo.")

    scan_days = st.slider("Días para el escaneo", 60, 365, 120, key="scan_days")
    scan_btn = st.button("🔭 Escanear todos los activos", type="primary")

    if scan_btn:
        results = []
        prog = st.progress(0)
        asset_list = list(ASSETS.items())

        for idx, (name, tk) in enumerate(asset_list):
            prog.progress((idx + 1) / len(asset_list), text=f"Analizando {name}…")
            df_s = fetch_data(tk, scan_days)
            if df_s.empty or len(df_s) < 40:
                continue
            df_s = calculate_squeeze_index(
                df_s, window, bb_mult, kc_mult, atr_period, threshold, use_spectrum, use_vol_filter
            )
            if len(df_s) < 10:
                continue
            last_s = df_s.iloc[-1]
            bt_s = run_backtest(df_s, forward_days=5)
            prec_s = (bt_s["Acierto"] == "✅").sum() / len(bt_s) * 100 if not bt_s.empty else None
            num_ep = int(df_s["SqueezeEpisode"].max())

            results.append({
                "Activo": name,
                "SqueezeOn": "🟢" if last_s["SqueezeOn"] else "⚪",
                "Señal": "🚨 FUERTE" if last_s["SqueezeDetected"] else (
                         "⏳ LEVE" if last_s["SqueezeOn"] else "—"),
                "SI": round(last_s["SqueezeIndex"], 1),
                "Trend": round(last_s["Trend"], 3),
                "Dirección": last_s["Direction"],
                "Episodios": num_ep,
                "Precisión_bt": f"{prec_s:.0f}%" if prec_s is not None else "—",
                "Lambda_Ω": round(last_s["Lambda"], 1),
                "Precio": round(last_s["Close"], 4),
            })

        prog.empty()

        if results:
            df_res = pd.DataFrame(results).sort_values("SI", ascending=False)
            st.dataframe(df_res, use_container_width=True, height=420)

            # Email
            st.divider()
            user_email = st.text_input("📧 Enviar reporte a:", placeholder="tu@email.com", key="scan_email")
            if st.button("Enviar reporte", key="scan_send"):
                signals_str = "\n".join(
                    f"• {r['Activo']} | {r['Señal']} | {r['Dirección']} | SI {r['SI']} | Precisión {r['Precisión_bt']}"
                    for _, r in df_res.iterrows() if r["SqueezeOn"] != "⚪"
                ) or "Sin señales activas"
                body = (
                    f"SqueezeIndex v2.1 — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"Parámetros: window={window} bb={bb_mult} kc={kc_mult} threshold={threshold}\n\n"
                    f"=== SEÑALES ACTIVAS ===\n{signals_str}\n"
                )
                ok, msg = send_email_resend(user_email, f"Squeeze Scan {datetime.now(UTC).date()}", body)
                st.success("Enviado ✅") if ok else st.error(msg)
        else:
            st.warning("No se obtuvieron resultados del escaneo.")

            # ══════════════════════════════════════════════════════════════════════════════
# TAB METODOLOGÍA — Totalmente adaptada al código unificado v2.1
# (Glosario y explicaciones actualizadas según los nuevos cálculos: Welch, Lorentzian, 4 componentes, episodios, backtest post-episodio)
# ══════════════════════════════════════════════════════════════════════════════

with tab_metodologia:
    st.markdown("""
    ## 〰️ Metodología — SqueezeIndex v2.1 (Best of Both)

    ### El modelo como física de ondas

    El precio de un activo financiero se trata como una **señal ondulatoria** que, como las olas del mar,
    alterna entre estados de alta energía (volatilidad) y baja energía (compresión). Cuando la energía
    se acumula sin liberarse, el sistema se comporta como un **oscilador subamortiguado** cargado:
    la liberación posterior es proporcional a la energía almacenada.

    Este principio físico sigue siendo el corazón del modelo, pero ahora calculado con mayor rigor matemático.

    ---
    """)

    with st.expander("📖 Glosario completo (actualizado a v2.1)", expanded=True):
        st.markdown("""
        **Conceptos básicos (explicados para principiantes, adaptados al nuevo código):**

        - **Compresión (Squeeze)**  
          Es cuando el precio deja de moverse fuerte y se queda “encerrado” en un rango muy estrecho durante **varios días consecutivos**.  
          *Analogía del mar*: El agua está tan plana que parece que no pasa nada… pero la energía se está acumulando debajo.  
          *En v2.1*: Solo se considera compresión real cuando dura **mínimo 3 días seguidos** (episodio). Esto elimina ruido de 1-2 días.

        - **SqueezeOn (la luz verde)**  
          Es la señal clara que dice: “¡En este momento el precio está realmente comprimido!”  
          Aparece como fondo verde en el gráfico principal.  
          *Analogía*: Es el semáforo en verde que te avisa “el resorte ya está muy tenso, prepárate”.  
          *En v2.1*: Se activa cuando las Bandas de Bollinger entran completamente dentro del Canal de Keltner.

        - **SqueezeIndex (actualizado)**  
          Nuestro medidor principal de “cuánta tensión hay acumulada”.  
          **Cómo se calcula ahora (v2.1)**:  
          1. Se mide qué tan estrecho está el BB respecto a su propio historial (percentil).  
          2. Se multiplica por un factor de calidad de Lambda (Lorentzian).  
          3. Se normaliza a escala 0-100.  
          *Analogía*: Es el velocímetro del resorte, pero ahora calibrado para que 80-90 signifique “extremadamente cargado” independientemente del activo.  
          *Qué pasa si el índice es muy alto (>80)*: La probabilidad de un movimiento grande aumenta significativamente.

        - **Lambda (Λ) — Longitud de onda dominante (Welch PSD)**  
          Mide el “ritmo natural” de ese activo concreto.  
          **Cómo se calcula ahora (v2.1)**: Usamos el método **Welch** (estimador espectral de potencia). Divide la ventana en segmentos, calcula la FFT de cada uno y encuentra la frecuencia con más potencia.  
          *Por qué Welch y no conteo de picos*: Es mucho más resistente al ruido y detecta la periodicidad estadísticamente dominante, no solo la más reciente.  
          *Analogía*: Cada activo tiene su propio “latido”. El oro late despacio, el Bitcoin late rápido. Lambda lo detecta automáticamente.

        - **Trend (Tendencia) — 4 componentes**  
          Te dice hacia dónde es más probable que salte el precio cuando termine la compresión.  
          **Cómo se calcula ahora (v2.1)**:  
          - 35% Slope largo (ventana = Lambda)  
          - 25% Slope corto (6 barras)  
          - 25% MFI (presión compradora/vendedora)  
          - 15% ROC normalizado por ATR  
          Todo normalizado con `tanh` para controlar valores extremos.  
          *Analogía del mar*: Es como leer la dirección del viento + la fuerza de la corriente + la presión del agua antes de que llegue la ola grande.

        - **SqueezeDetected (la señal más poderosa)**  
          Se enciende solo cuando hay **mucha compresión + una dirección clara + régimen de volatilidad baja**.  
          *En v2.1*: Además del Trend > threshold, aplicamos un filtro que ignora señales cuando el ATR está en percentil alto (mercados ya muy nerviosos).  
          *Qué pasa cuando se activa*: Estadísticamente, el precio tiende a moverse con fuerza en la dirección del Trend en los días siguientes.

        **Parámetros del modelo (y qué pasa si los cambias) — actualizado v2.1:**

        - **Ventana = 20 días** (por defecto)  
          Es la ventana principal para BB, EMA y cálculo de Lambda.  
          *Qué pasa si la subes a 50 días*: Detecta compresiones más largas y “históricas”.  
          *Qué pasa si la bajas a 10 días*: Reacciona más rápido, pero también da más señales falsas (ruido).

        - **Bandas de Bollinger (BB Mult = 2.0)**  
          Definen el rango “normal” del precio.  
          *Qué pasa si subes a 2.5*: Detectas menos squeezes (más conservador).  
          *Qué pasa si bajas a 1.5*: Detectas más squeezes, pero también más falsos (más agresivo).

        - **Canales de Keltner (KC Mult = 1.5)**  
          Definen el túnel de volatilidad real.  
          SqueezeOn se activa cuando BB entra completamente dentro de KC.

        - **ATR Period = 20**  
          Mide la volatilidad real. Se usa tanto para Keltner como para normalizar el Trend.

        - **Trend Threshold = 0.15**  
          Nivel mínimo de fuerza de dirección para activar SqueezeDetected.  
          *Qué pasa si lo subes a 0.25*: Solo las señales más fuertes y claras.  
          *Qué pasa si lo bajas a 0.05*: Más señales, pero más ruido.

        - **Filtro de régimen de volatilidad (nuevo en v2.1)**  
          Solo se activan señales cuando el ATR está por debajo del percentil 75.  
          Evita operar en mercados ya muy volátiles donde el “squeeze” pierde significado.
        """)

    st.markdown("""
    ---

    ### Mejoras técnicas v2.1 (por qué es mejor que la versión anterior)

    | Componente              | Versión anterior          | v2.1 (código unificado)                              | Beneficio real |
    |-------------------------|---------------------------|-------------------------------------------------------|---------------|
    | **Lambda**              | Conteo simple de picos    | Welch PSD vectorizado                                 | Mucho más robusto al ruido |
    | **SqueezeIndex**        | 1/BBWidth × 1/Lambda      | Percentil relativo + Lorentzian quality               | Comparable entre activos + mejor manejo de colas |
    | **Trend**               | 3 componentes             | 4 componentes (slope largo + corto + MFI + ROC) + tanh| Mayor estabilidad estadística |
    | **Episodios**           | No existían               | Detección automática de ≥3 días consecutivos          | Elimina falsos squeezes de 1-2 días |
    | **Backtest**            | Durante la compresión     | Retorno **post-episodio** (N días después)            | Mide el verdadero poder predictivo |
    | **Señal**               | Binaria                   | SignalStrength (0-1) + filtro de régimen de vol       | Más selectiva y segura |

    **Principio fundamental (estilo Jim Simons):**  
    Nunca midas si el modelo “acierta” durante la compresión.  
    La única medición válida es el retorno **después** de que el episodio de compresión termine.

    Este modelo no promete ganancias. Solo te avisa, con rigor matemático y explicaciones claras, cuándo el mercado está “cargado” y listo para moverse.

    ---
    """)

    st.success("✅ App versión 2.1 — Best of Both: Metodología clara + Edge cuantitativo real unificado")
