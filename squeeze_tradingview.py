import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta, UTC
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Squeezeindex", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.title("Squeezeindex")
st.caption(f"Última ejecución: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ===================== PESTAÑAS =====================
tab_dashboard, tab_info = st.tabs(["📈 Dashboard", "🧠 Info"])

with tab_info:
    # ===================== EXPLICACIÓN INICIAL =====================
    st.markdown("""
### ¿Qué es el Squeezeindex?

Imagina que el precio de una acción, del oro, del petróleo o del Bitcoin es como **el mar**.
A veces hay olas grandes y el agua se mueve mucho.

Otras veces el mar se queda **casi plano y en calma total** durante muchos días seguidos.
A esa calma extrema la llamamos **compresión** o “Squeeze”.

Este modelo:
1. Detecta cuándo el precio está en una compresión fuerte (el resorte está muy apretado).
2. Mide cuánta energía se ha acumulado.
3. Intenta estimar hacia dónde es más probable que salte el precio cuando se libere.
""")
    with st.expander("📖 Glosario)", expanded=True):
        st.markdown("""
    **Conceptos:**
    - **Compresión (Squeeze)**
      Cuando el precio deja de moverse fuerte y se queda “encerrado” en un rango muy estrecho durante varios días.
      *Analogía*: Es como apretar un resorte o contener la respiración. Todo está muy quieto, pero la tensión aumenta.
      
    - **SqueezeOn (la luz verde)**
      Es la señal que dice: “¡En este momento el precio está realmente comprimido!”
      Aparece como fondo verde en el gráfico principal.
      *Analogía*: La luz del semáforo que se pone en verde para avisarte que el resorte está muy tenso.
      
    - **SqueezeIndex**
      Nuestro medidor principal de “cuánta tensión hay”.
      Cuanto **más alto** es el número, más energía se está acumulando.
      *Analogía*: Es como el velocímetro del resorte: te dice qué tan fuerte está apretado.
      
    - **Lambda (Λ)**
      Mide el “ritmo natural” del precio (cuánto tarda normalmente en hacer una pequeña subida y bajada).
      *Analogía*: Cada persona camina a su propio paso. Lambda detecta el “paso” de cada activo (el oro camina distinto que el Bitcoin) para que las matemáticas se adapten perfectamente.
    
    - **Trend (Tendencia)**
      Te dice hacia dónde es más probable que salte el precio cuando termine la compresión:
      - Arriba → **Alcista** (positivo)
      - Abajo → **Bajista** (negativo)
      *Analogía*: Es como leer la dirección del viento antes de que llegue la ola grande.
   
    - **SqueezeDetected**
      La **señal más poderosa** del modelo.
      Se enciende solo cuando hay **mucha compresión + una dirección clara**.
      Estas son las situaciones que más nos interesan para prestar atención.
    
    
    **Parámetros internos del modelo:**
   
    - **Ventana = 20 días**
      El modelo mira los últimos 20 días para entender cómo se está comportando el precio.
      (Es como usar una foto reciente del mercado).
    
    - **Bandas de Bollinger (BB Mult = 2.0)**
      Son dos bandas azules que marcan el rango “normal” del precio. Cuando se estrechan mucho, indican calma.
    
    - **Canales de Keltner (KC Mult = 1.5)**
      Otro túnel más ajustado que usa la volatilidad real. Cuando las bandas azules entran completamente dentro de este túnel → SqueezeOn.
    
    - **ATR Period = 20**
      Mide cuánto se mueve normalmente el precio cada día (la volatilidad real).
    
    - **Trend Threshold = 0.15**
      Nivel mínimo de fuerza de dirección que necesitamos para decir “este Squeeze tiene una tendencia clara”.
    """)

# ===================== API KEY =====================
API_KEY = st.secrets["API_KEY"]

# ===================== BARRA LATERAL =====================
with st.sidebar:
    st.header("🎛️ Control de Ondas")
    assets = {"EURUSD": "C:EURUSD", "Oro": "C:XAUUSD", "Plata": "C:XAGUSD",
              "SPY": "SPY", "GBPUSD": "C:GBPUSD", "USDJPY": "C:USDJPY",
              "BTCUSD": "X:BTCUSD", "USO": "USO"}
    selected_asset = st.selectbox("Activo", options=list(assets.keys()))
    ticker = assets[selected_asset]
   
    days_slider = st.slider("Días de datos (máx free tier)", 30, 730, 365)
    days = st.number_input("O escribe el número exacto de días", min_value=30, max_value=730, value=days_slider, step=1)
   
    st.subheader("⚙️ Parámetros del SqueezeIndex")
    window = st.slider("Window (EMA / BB / Lambda)", 10, 60, 20)
    bb_mult = st.slider("Bollinger Multiplier", 1.0, 3.5, 2.0, 0.1)
    kc_mult = st.slider("Keltner Multiplier", 1.0, 3.0, 1.5, 0.1)
    atr_period = st.slider("ATR Period", 10, 40, 20)
    trend_threshold = st.slider("Trend Threshold (para SqueezeDetected)", 0.05, 0.40, 0.15, 0.01)
   
    update_button = st.button("🔄 Actualizar datos y calcular SqueezeIndex", type="primary", use_container_width=True)

# ===================== FUNCIONES =====================
@st.cache_data(ttl=3600)
def fetch_data(ticker, days=730):
    now = datetime.now(UTC)
    end_date = (now + timedelta(days=1)).date()
    start_date = end_date - timedelta(days=days)
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'results' not in data or not data['results']:
            st.error(f"No hay datos para {ticker}")
            return pd.DataFrame()
        df = pd.DataFrame(data['results'])
        df['Date'] = pd.to_datetime(df['t'], unit='ms').dt.date
        df['Open'] = df.get('o', df['c'])
        df['High'] = df['h']
        df['Low'] = df['l']
        df['Close'] = df['c']
        df = df[['Date', 'Open', 'High', 'Low', 'Close']].sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error al descargar {ticker}: {str(e)[:100]}")
        return pd.DataFrame()

def calculate_squeeze_index(df, window, bb_mult, kc_mult, atr_period, trend_threshold):
    if df.empty or len(df) < 30:
        return df
    df["EMA"] = df["Close"].ewm(span=window, adjust=False).mean()
    df["STD"] = df["Close"].rolling(window).std()
    df["UpperBB"] = df["EMA"] + bb_mult * df["STD"]
    df["LowerBB"] = df["EMA"] - bb_mult * df["STD"]
    df["BBWidth"] = (df["UpperBB"] - df["LowerBB"]) / df["EMA"]
   
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['PrevClose']), abs(df['Low'] - df['PrevClose'])], axis=1).max(axis=1).fillna(0)
    df['ATR'] = df['TR'].ewm(span=atr_period, adjust=False).mean()
   
    df["UpperKC"] = df["EMA"] + kc_mult * df['ATR']
    df["LowerKC"] = df["EMA"] - kc_mult * df['ATR']
   
    smoothed = df["Close"].ewm(span=5, adjust=False).mean()
    df["Lambda"] = np.nan
    for i in range(window - 1, len(df)):
        prices = smoothed.iloc[i - window + 1:i + 1].values
        peaks, _ = find_peaks(prices)
        valleys, _ = find_peaks(-prices)
        extrema = np.sort(np.concatenate([peaks, valleys]))
        if len(extrema) > 1:
            df.loc[i, "Lambda"] = np.mean(np.diff(extrema))
    df["Lambda"] = df["Lambda"].ffill().fillna(5.0)
    df["SqueezeIndex"] = (1 / df["BBWidth"]) * (1 / df["Lambda"])
   
    df["SqueezeOn"] = (df["UpperBB"] <= df["UpperKC"]) & (df["LowerBB"] >= df["LowerKC"])
   
    # Trend ultra-reactivo
    df["Trend_slope"] = np.nan
    for i in range(len(df)):
        lam = df["Lambda"].iloc[i] if not np.isnan(df["Lambda"].iloc[i]) else 10
        w = max(6, int(lam * 1.0))
        if i < w - 1: continue
        x = np.arange(w)
        y = smoothed.iloc[i - w + 1:i + 1].values
        slope = np.polyfit(x, y, 1)[0]
        atr_val = df["ATR"].iloc[i]
        df.loc[i, "Trend_slope"] = slope / atr_val if atr_val != 0 else 0
    slope_norm = np.tanh(df["Trend_slope"].fillna(0) * 5)
   
    mid = (df['High'] + df['Low']) / 2
    range_ = (df['High'] - df['Low']).replace(0, np.nan)
    bull_pressure = ((df['Close'] - mid) / range_).clip(-1, 1).fillna(0)
    raw_flow = bull_pressure * range_
    pos_flow = raw_flow.clip(lower=0).rolling(14).sum()
    neg_flow = (-raw_flow).clip(lower=0).rolling(14).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        mfi = np.where(neg_flow != 0, 100 - 100 / (1 + pos_flow / neg_flow), 50.0)
    mfi_score = ((pd.Series(mfi, index=df.index) - 50) / 50).clip(-1, 1)
   
    df["Short_Slope"] = np.nan
    short_w = 6
    for i in range(short_w, len(df)):
        x_short = np.arange(short_w)
        y_short = smoothed.iloc[i - short_w + 1:i + 1].values
        slope_short = np.polyfit(x_short, y_short, 1)[0]
        atr_val = df["ATR"].iloc[i]
        df.loc[i, "Short_Slope"] = slope_short / atr_val if atr_val != 0 else 0
    short_norm = np.tanh(df["Short_Slope"].fillna(0) * 5)
   
    df["Trend"] = (0.50 * slope_norm + 0.30 * mfi_score + 0.20 * short_norm).ewm(span=2, adjust=False).mean()
    df["Direction"] = np.where(df["Trend"] > 0, "Alcista", np.where(df["Trend"] < 0, "Bajista", "Neutral"))
    df["SqueezeDetected"] = df["SqueezeOn"] & (abs(df["Trend"]) > trend_threshold)
    return df

def send_email_resend(to_email, subject, body):
    RESEND_API_KEY = st.secrets["RESEND_API_KEY"]
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "from": "Squeeze Report <onboarding@resend.dev>",
        "to": [to_email],
        "subject": subject,
        "text": body
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            return True, "Email enviado correctamente"
        else:
            return False, f"Error Resend: {response.text[:150]}"
    except Exception as e:
        return False, f"Error enviando email: {str(e)[:100]}"

# ===================== LÓGICA PRINCIPAL (solo en Dashboard) =====================
with tab_dashboard:
    if update_button:
        with st.spinner(f"Descargando {selected_asset} ({days} días)..."):
            df_raw = fetch_data(ticker, days)
            if not df_raw.empty:
                df = calculate_squeeze_index(df_raw.copy(), window, bb_mult, kc_mult, atr_period, trend_threshold)
               
                if len(df) < 30 or 'SqueezeIndex' not in df.columns:
                    st.warning(f"⚠️ Solo {len(df)} velas. Prueba con más días.")
                else:
                    st.success(f"✅ {selected_asset} cargado: {len(df)} velas | Última: {df['Date'].iloc[-1]}")
                   
                    last = df.iloc[-1]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("SqueezeIndex", f"{last['SqueezeIndex']:.2f}")
                    col2.metric("Trend", f"{last['Trend']:.3f} ({last['Direction']})")
                    col3.metric("Estado", "🟢 SqueezeOn" if last['SqueezeOn'] else "🔴 Sin compresión")
                   
                    # Estadísticas
                    squeeze_days = df['SqueezeOn'].sum()
                    detected_days = df['SqueezeDetected'].sum()
                    avg_squeeze = df.loc[df['SqueezeOn'], 'SqueezeIndex'].mean() if squeeze_days > 0 else 0
                    pct_squeeze = (squeeze_days / len(df)) * 100
                   
                    st.subheader("📊 Estadísticas del Squeezeindex")
                    colA, colB, colC, colD = st.columns(4)
                    colA.metric("Días en compresión", f"{squeeze_days}")
                    colB.metric("Promedio SqueezeIndex", f"{avg_squeeze:.2f}")
                    colC.metric("Explosiones detectadas", f"{detected_days}")
                    colD.metric("% tiempo en SqueezeOn", f"{pct_squeeze:.1f}%")
                   
                    # Gráfico
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.20],
                                        subplot_titles=("Precio + Bollinger + Keltner + Fondo SqueezeOn",
                                                        "SqueezeIndex (compresión de onda)",
                                                        "Trend Ultra-Reactivo + Señales de Explosión"))
                   
                    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Close'].shift(1).fillna(df['Close']),
                                                 high=df['High'], low=df['Low'], close=df['Close'], name="Precio"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['UpperBB'], line=dict(color='blue'), name='Upper BB'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['LowerBB'], line=dict(color='blue'), fill='tonexty', fillcolor='rgba(0,0,255,0.05)'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['UpperKC'], line=dict(color='orange', dash='dash'), name='Upper KC'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['LowerKC'], line=dict(color='orange', dash='dash'), name='Lower KC'), row=1, col=1)
                   
                    for i in range(len(df)-1):
                        if df['SqueezeOn'].iloc[i]:
                            fig.add_vrect(x0=df['Date'].iloc[i], x1=df['Date'].iloc[i+1], fillcolor="green", opacity=0.18, layer="below", line_width=0, row=1, col=1)
                   
                    colors = np.where(df['SqueezeIndex'] > df['SqueezeIndex'].mean()*1.5, 'red', 'orange')
                    fig.add_trace(go.Bar(x=df['Date'], y=df['SqueezeIndex'], marker_color=colors), row=2, col=1)
                   
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['Trend'], mode='lines+markers',
                                             marker=dict(color=np.where(df['SqueezeDetected'], 'gold', 'blue'),
                                                         size=np.where(df['SqueezeDetected'], 8, 4))), row=3, col=1)
                   
                    fig.update_xaxes(title_text="Fecha", showgrid=True, gridcolor="rgba(128,128,128,0.3)")
                    fig.update_yaxes(title_text="Precio", showgrid=True, gridcolor="rgba(128,128,128,0.3)", tickformat=",.2f", row=1, col=1)
                    fig.update_yaxes(title_text="SqueezeIndex", showgrid=True, gridcolor="rgba(128,128,128,0.3)", row=2, col=1)
                    fig.update_yaxes(title_text="Trend", showgrid=True, gridcolor="rgba(128,128,128,0.3)", row=3, col=1)
                   
                    fig.update_layout(height=850, template="plotly_dark", hovermode="x unified",
                                      title=f"{selected_asset} — Ondas de Compresión y Explosión")
                    st.plotly_chart(fig, use_container_width=True)
                   
                    # Expander
                    with st.expander("🔎 Ver TODAS las detecciones SqueezeOn del período completo", expanded=True):
                        squeeze_events = df[df['SqueezeOn'] == True].copy()
                        if not squeeze_events.empty:
                            squeeze_events['Prev_Close'] = squeeze_events['Close'].shift(1)
                            squeeze_events['Variacion_Diaria_%'] = ((squeeze_events['Close'] - squeeze_events['Prev_Close']) / squeeze_events['Prev_Close']) * 100
                            squeeze_events['Direccion_Diaria'] = np.where(squeeze_events['Variacion_Diaria_%'] > 0, "Alcista",
                                                                          np.where(squeeze_events['Variacion_Diaria_%'] < 0, "Bajista", "Neutral"))
                            squeeze_events['Squeeze_Acierta'] = np.where(squeeze_events['Direccion_Diaria'] == squeeze_events['Direction'], "✅ Acierta", "❌ Incorrecto")
                           
                            tabla = squeeze_events[['Date', 'Close', 'Variacion_Diaria_%', 'Direccion_Diaria',
                                                    'Trend', 'Direction', 'Squeeze_Acierta']].round(3)
                            st.dataframe(tabla, use_container_width=True)
                           
                            aciertos = (squeeze_events['Squeeze_Acierta'] == "✅ Acierta").sum()
                            total = len(squeeze_events)
                            precision = (aciertos / total) * 100 if total > 0 else 0
                            st.metric("Precisión histórica del Squeeze", f"{precision:.1f}%", f"{aciertos}/{total} aciertos")
                        else:
                            st.info("No se detectaron SqueezeOn en todo el rango.")

    else:
        st.info("👈 Elige activo, ajusta parámetros en la barra lateral y pulsa 'Actualizar datos'")

    # ===================== BOTÓN EMAIL =====================
    st.divider()
    st.subheader("📧 Enviar Reporte por Email")
    user_email = st.text_input("Correo donde quieres recibir el reporte", placeholder="tu@email.com")
    if st.button("📤 Enviar Reporte (Solo Detecciones Leve + Fuerte)", type="secondary"):
        if not user_email:
            st.warning("Por favor escribe un correo electrónico.")
        else:
            with st.spinner("Escaneando todos los activos..."):
                assets = {"EURUSD": "C:EURUSD", "Oro": "C:XAUUSD", "Plata": "C:XAGUSD", "SPY": "SPY",
                          "GBPUSD": "C:GBPUSD", "USDJPY": "C:USDJPY", "BTCUSD": "X:BTCUSD", "USO": "USO"}
               
                signals = []
                compression = []
               
                for asset, ticker in assets.items():
                    df_raw = fetch_data(ticker, days)
                    if df_raw.empty or len(df_raw) < 30:
                        continue
                    df = calculate_squeeze_index(df_raw.copy(), window, bb_mult, kc_mult, atr_period, trend_threshold)
                    if len(df) < 30:
                        continue
                    last = df.iloc[-1]
                    if not (last['SqueezeOn'] or last['SqueezeDetected']):
                        continue
                   
                    info = {
                        "asset": asset,
                        "squeeze_index": round(last["SqueezeIndex"], 2),
                        "trend": round(last["Trend"], 4),
                        "direction": last["Direction"],
                        "price": round(last["Close"], 2)
                    }
                    if last['SqueezeDetected']:
                        signals.append(info)
                    elif last['SqueezeOn']:
                        compression.append(info)
               
                subject = f"Squeeze Report - {datetime.now(UTC).strftime('%Y-%m-%d')}"
                body = f"Reporte generado: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                body += f"Parámetros: window={window}, bb_mult={bb_mult}, kc_mult={kc_mult}, threshold={trend_threshold}\n\n"
               
                if signals:
                    body += "=== SEÑALES FUERTES ===\n"
                    for a in signals:
                        body += f"• {a['asset']} | {a['direction']} | SqueezeIndex {a['squeeze_index']} | Trend {a['trend']}\n"
                if compression:
                    body += "\n=== DETECCIONES LEVES ===\n"
                    for c in compression:
                        body += f"• {c['asset']} | {c['direction']} | SqueezeIndex {c['squeeze_index']} | Trend {c['trend']}\n"
               
                success, msg = send_email_resend(user_email, subject, body)
                if success:
                    st.success(f"✅ Reporte enviado a {user_email}")
                else:
                    st.error(msg)

st.success("✅ App versión 1.9 — Dashboard limpio + pestaña Info con tus definiciones exactas")
