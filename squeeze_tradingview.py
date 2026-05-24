import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta, UTC
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Squeeze Wave Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.title("🌀 Squeeze Wave TradingView")
st.markdown("**Tu propio TradingView privado** — Matemática de ondas + SqueezeIndex + Compresión explosiva")
st.caption(f"Última ejecución: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ===================== EXPLICACIÓN INICIAL=====================
st.markdown("""
### ¿Qué es el Squeeze Wave Model? 

Imagina que el precio de una acción, del oro, del petróleo o del Bitcoin es como **el mar**.

A veces hay olas grandes y el agua se mueve mucho.  
Otras veces el mar se queda **casi plano y en calma total** durante muchos días seguidos.

A esa calma extrema la llamamos **compresión** o “Squeeze”.  
Es como cuando aprietas un resorte (muelle) con las dos manos:  
- Todo está muy quieto…  
- pero la energía se va acumulando poco a poco dentro del resorte.

Cuando el resorte ya no aguanta más, **¡salta con fuerza** en una dirección.

Este modelo matemático es exactamente ese radar inteligente:  
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
st.divider()

# ===================== API KEY =====================
API_KEY = st.secrets["API_KEY"]

# ===================== BARRA LATERAL (PARÁMETROS CONFIGURABLES) =====================
with st.sidebar:
    st.header("🎛️ Control de Ondas")
    
    # Activos
    assets = {"EURUSD": "C:EURUSD", "Oro (XAUUSD)": "C:XAUUSD", "Plata (XAGUSD)": "C:XAGUSD",
              "SPY": "SPY", "GBPUSD": "C:GBPUSD", "USDJPY": "C:USDJPY",
              "BTCUSD": "X:BTCUSD", "USO (Petróleo)": "USO"}
    selected_asset = st.selectbox("Activo", options=list(assets.keys()))
    ticker = assets[selected_asset]
    
    # Días de datos
    days_slider = st.slider("Días de datos (máx free tier)", 30, 730, 365)
    days = st.number_input("O escribe el número exacto de días", min_value=30, max_value=730, value=days_slider, step=1)
    
    # === PARÁMETROS CONFIGURABLES ===
    st.subheader("⚙️ Parámetros Matemáticos")
    window = st.slider("Ventana (días)", 10, 50, 20)
    bb_mult = st.slider("BB Multiplicador", 1.0, 3.0, 2.0, step=0.1)
    kc_mult = st.slider("KC Multiplicador", 1.0, 3.0, 1.5, step=0.1)
    atr_period = st.slider("ATR Period", 10, 50, 20)
    trend_threshold = st.slider("Trend Threshold", 0.05, 0.50, 0.15, step=0.05)
    
    update_button = st.button("🔄 Actualizar datos y calcular SqueezeIndex", type="primary", use_container_width=True)

# ===================== FUNCIONES (sin cambios) =====================
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

def calculate_squeeze_index(df, window=20, bb_mult=2.0, kc_mult=1.5, atr_period=20, trend_threshold=0.15):
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

# ===================== FUNCIONALIDAD DE EMAIL =====================
def send_squeeze_alert(email_to, asset, squeeze_index, trend, direction):
    if not st.secrets.get("EMAIL_USER") or not st.secrets.get("EMAIL_PASS"):
        st.warning("Configura EMAIL_USER y EMAIL_PASS en secrets para activar alertas por email.")
        return
    try:
        msg = MIMEMultipart()
        msg['From'] = st.secrets["EMAIL_USER"]
        msg['To'] = email_to
        msg['Subject'] = f"🚨 Squeeze Wave Alert: {asset} listo para explotar"
        
        body = f"""
        Squeeze detectado en {asset}!
        SqueezeIndex: {squeeze_index:.2f}
        Trend: {trend:.3f} ({direction})
        Momento de alta probabilidad de movimiento fuerte.
        """
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
        server.send_message(msg)
        server.quit()
        st.success(f"✅ Alerta enviada a {email_to}")
    except Exception as e:
        st.error(f"Error enviando email: {str(e)}")

# ===================== LÓGICA PRINCIPAL =====================
if update_button:
    with st.spinner(f"Descargando {selected_asset} ({days} días)..."):
        df_raw = fetch_data(ticker, days)
        if not df_raw.empty:
            df = calculate_squeeze_index(df_raw.copy(), window, bb_mult, kc_mult, atr_period, trend_threshold)
            
            if len(df) < 30 or 'SqueezeIndex' not in df.columns:
                st.warning(f"⚠️ Solo {len(df)} velas. Prueba con más días.")
                st.dataframe(df_raw.tail(10), use_container_width=True)
            else:
                st.success(f"✅ {selected_asset} cargado: {len(df)} velas | Última: {df['Date'].iloc[-1]}")
                
                last = df.iloc[-1]
                col1, col2, col3 = st.columns(3)
                col1.metric("SqueezeIndex", f"{last['SqueezeIndex']:.2f}")
                col2.metric("Trend", f"{last['Trend']:.3f} ({last['Direction']})")
                col3.metric("Estado", "🟢 SqueezeOn" if last['SqueezeOn'] else "🔴 Sin compresión")
                
                # Estadísticas...
                
                # === EMAIL ALERT (restaurado) ===
                if last['SqueezeDetected']:
                    st.subheader("🚨 Alerta de Explosión Detectada")
                    email_to = st.text_input("Enviar alerta por email a:", value=st.secrets.get("DEFAULT_EMAIL", ""))
                    if st.button("📧 Enviar alerta ahora", type="primary"):
                        send_squeeze_alert(email_to, selected_asset, last['SqueezeIndex'], last['Trend'], last['Direction'])

            

else:
    st.info("👈 Elige activo, ajusta parámetros y pulsa 'Actualizar datos'")

st.success("✅ App versión 1.8 — Parámetros configurables + Email restaurado + Explicación pedagógica")
