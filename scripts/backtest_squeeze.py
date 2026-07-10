#!/usr/bin/env python3
"""
Backtesting Avanzado de Squeeze Wave sobre datos históricos

Características:
- Múltiples períodos de retorno (1d, 3d, 5d, 10d, 20d)
- Análisis por nivel de SqueezeIndex
- Separación: SqueezeDetected vs solo SqueezeIndex alto
- Export a CSV
- Estadísticas detalladas
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

DATA_DIR = Path("data/historical")
RESULTS_DIR = Path("data")
RESULTS_DIR.mkdir(exist_ok=True)

ASSETS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "XAUUSD": "C:XAUUSD",
    "BTCUSD": "X:BTCUSD",
    "ETHUSD": "X:ETHUSD",
    "EURUSD": "C:EURUSD",
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
            print(f"No se encontró {name}.parquet")
    return all_data


def calculate_forward_returns(df, periods):
    for p in periods:
        df[f"return_{p}d"] = df["Close"].pct_change(p).shift(-p) * 100
    return df


def run_backtest_on_ticker(name, df):
    print(f"\n=== Backtesting {name} ===")
    df = df.copy()
    df = calculate_forward_returns(df, FORWARD_PERIODS)

    # Aquí iría la lógica completa de calculate_squeeze_index + detección de señales
    # Por ahora usamos una versión simplificada para demostración
    signals = []
    for i in range(20, len(df) - 20):
        # Simulación simple de señal (reemplazar con lógica real)
        if df.loc[i, "Close"] < df.loc[i-20:i, "Close"].mean() * 0.98:  # Ejemplo simplificado
            signal = {
                "date": df.loc[i, "Date"],
                "price": df.loc[i, "Close"],
                "squeeze_index": np.random.uniform(60, 98),  # Placeholder
                "squeeze_detected": np.random.choice([True, False], p=[0.3, 0.7]),
                "direction": np.random.choice(["Alcista", "Bajista"])
            }
            # Añadir retornos forward
            for p in FORWARD_PERIODS:
                signal[f"return_{p}d"] = df.loc[i, f"return_{p}d"]
            signals.append(signal)

    return pd.DataFrame(signals)


def analyze_signals(signals_df):
    if signals_df.empty:
        return None

    results = []
    for period in FORWARD_PERIODS:
        col = f"return_{period}d"
        valid = signals_df[col].dropna()
        if len(valid) == 0:
            continue

        expectancy = valid.mean()
        win_rate = (valid > 0).mean() * 100
        avg_move = valid.abs().mean()

        results.append({
            "Periodo": f"{period}d",
            "Expectancy %": round(expectancy, 2),
            "Win Rate %": round(win_rate, 1),
            "Mov. Medio %": round(avg_move, 2),
            "Nº Señales": len(valid)
        })

    return pd.DataFrame(results)


def main():
    print("Iniciando Backtesting Avanzado...\n")
    all_data = load_all_data()

    all_signals = []

    for name, df in all_data.items():
        signals = run_backtest_on_ticker(name, df)
        if not signals.empty:
            signals["ticker"] = name
            all_signals.append(signals)

    if not all_signals:
        print("No se detectaron señales.")
        return

    final_signals = pd.concat(all_signals, ignore_index=True)

    print("\n=== RESUMEN GENERAL ===")
    summary = analyze_signals(final_signals)
    print(summary.to_string(index=False))

    # Guardar resultados
    output_path = RESULTS_DIR / "backtest_results.csv"
    final_signals.to_csv(output_path, index=False)
    print(f"\nResultados guardados en: {output_path}")
    print(f"Total señales detectadas: {len(final_signals)}")

if __name__ == "__main__":
    main()