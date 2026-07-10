#!/usr/bin/env python3
"""
Backtesting Real de Squeeze Wave usando la lógica exacta de daily_alerts.py

Actualizado: listado de assets completado para alinearse con download_historical.py
y subscriptions.json. Universo expandido a 10 activos para mayor robustez estadística
en la estimación de expectancy post-compresión de ondas.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Importar funciones del script principal
try:
    from daily_alerts import calculate_squeeze_index, PARAMS, ALERT_MIN_SI, ALERT_MIN_STRENGTH
except ImportError:
    print("Error: No se pudo importar desde daily_alerts.py")
    exit(1)

DATA_DIR = Path("data/historical")
RESULTS_DIR = Path("data")
RESULTS_DIR.mkdir(exist_ok=True)

# LISTADO COMPLETO DE ASSETS
# Alineado con el sistema de producción (daily_alerts + subscriptions + download_historical)
# Incluye: 2 índices equity, 3 commodities (oro, plata, petróleo via ETF), 
# 3 pares forex majors, 2 cryptos.
# Esto permite testear la hipótesis de compresión de volatilidad (squeeze wave)
# a través de regímenes de mercado diferentes: equity trending, commodities con 
# ciclos estacionales/lambda distintos, forex mean-reverting, crypto alta frecuencia.
# Como en el enfoque de Jim Simons: sample size amplio + diversificación de 
# asset classes para separar señal de ruido y estimar edge realista.
ASSETS = {
    "SPY": "SPY",                    # S&P 500
    "QQQ": "QQQ",                    # Nasdaq-100
    "XAUUSD": "C:XAUUSD",            # Oro
    "XAGUSD": "C:XAGUSD",            # Plata
    "BTCUSD": "X:BTCUSD",            # Bitcoin
    "ETHUSD": "X:ETHUSD",            # Ethereum
    "EURUSD": "C:EURUSD",            # Euro Dólar
    "GBPUSD": "C:GBPUSD",            # Libra Dólar
    "USDJPY": "C:USDJPY",            # Yen Dólar
    "USO": "USO",                    # Petróleo (proxy WTI via ETF)
}

FORWARD_PERIODS = [1, 3, 5, 10, 20]


def load_all_data():
    all_data = {}
    for name in ASSETS.keys():
        filepath = DATA_DIR / f"{name}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            df = df.sort_values("Date").reset_index(drop=True)
            all_data[name] = df
            print(f"Cargado {name}: {len(df)} filas")
        else:
            print(f"No se encontró archivo para {name}")
    return all_data


def calculate_forward_returns(df):
    for p in FORWARD_PERIODS:
        df[f"fwd_return_{p}d"] = df["Close"].pct_change(p).shift(-p) * 100
    return df


def detect_signals(df):
    """Detecta señales usando la misma lógica que daily_alerts.py (ANY de las 3 condiciones)"""
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        si = row.get("SqueezeIndex", 0)
        strength = row.get("SignalStrength", 0)
        detected = row.get("SqueezeDetected", False)

        if detected or si >= ALERT_MIN_SI or strength >= ALERT_MIN_STRENGTH:
            signal = {
                "date": row["Date"],
                "ticker": row.get("ticker", ""),
                "price": row["Close"],
                "squeeze_index": si,
                "signal_strength": strength,
                "squeeze_detected": detected,
                "direction": row.get("Direction", ""),
                "trend": row.get("Trend", 0)
            }
            # Añadir retornos forward
            for p in FORWARD_PERIODS:
                col = f"fwd_return_{p}d"
                if col in df.columns:
                    signal[col] = row[col]
            signals.append(signal)
    return pd.DataFrame(signals)


def analyze_by_level(signals_df):
    if signals_df.empty:
        return pd.DataFrame()

    levels = [
        (0, 60, "Bajo"),
        (60, 75, "Medio"),
        (75, 90, "Alto"),
        (90, 200, "Extremo")
    ]

    results = []
    for low, high, label in levels:
        mask = (signals_df["squeeze_index"] >= low) & (signals_df["squeeze_index"] < high)
        level_signals = signals_df[mask]
        if level_signals.empty:
            continue

        row_data = {"Nivel": label, "Señales": len(level_signals)}
        for p in FORWARD_PERIODS:
            col = f"fwd_return_{p}d"
            if col in level_signals.columns:
                valid = level_signals[col].dropna()
                if len(valid) > 0:
                    row_data[f"Exp_{p}d %"] = round(valid.mean(), 2)
                    row_data[f"Win_{p}d %"] = round((valid > 0).mean() * 100, 1)
        results.append(row_data)

    return pd.DataFrame(results)


def main():
    print("=== BACKTESTING REAL CON LÓGICA DE PRODUCCIÓN ===\n")
    all_data = load_all_data()

    all_signals_list = []

    for name, df in all_data.items():
        print(f"\nProcesando {name}...")
        df = calculate_forward_returns(df)

        # Ejecutar cálculo real de SqueezeIndex
        df = calculate_squeeze_index(df, **PARAMS)

        # Detectar señales con la lógica real
        signals = detect_signals(df)
        if not signals.empty:
            signals["ticker"] = name
            all_signals_list.append(signals)
            print(f"  Señales detectadas: {len(signals)}")

    if not all_signals_list:
        print("\nNo se detectaron señales en el histórico.")
        return

    final_signals = pd.concat(all_signals_list, ignore_index=True)

    print("\n=== RESUMEN POR NIVEL DE SQUEEZEINDEX ===")
    summary = analyze_by_level(final_signals)
    print(summary.to_string(index=False))

    # Guardar resultados detallados
    output_file = RESULTS_DIR / "backtest_results.csv"
    final_signals.to_csv(output_file, index=False)
    print(f"\nResultados detallados guardados en: {output_file}")
    print(f"Total señales históricas detectadas: {len(final_signals)}")

if __name__ == "__main__":
    main()
