# SqueezeIndex v3.0

**Detección de compresión de volatilidad mediante análisis de ondas y espectral**

Sistema cuantitativo que modela el precio como señal ondulatoria para identificar períodos de **compresión continua de volatilidad** (squeezes) que estadísticamente preceden movimientos abruptos en activos financieros (forex, commodities, índices y cripto).

El enfoque es pragmático: detectar regímenes donde se acumula "energía" durante fases de baja amplitud, de forma que la magnitud de la expansión posterior tienda a ser proporcional a la compresión observada + calidad del ciclo dominante.

## Estado del proyecto
- App Streamlit completa y operativa (`squeeze_tradingview.py`)
- Backtest riguroso por episodios con métricas reales (expectancy, profit factor, Sharpe, Calmar, etc.)
- Alertas diarias automatizadas por email (8:00 AM CEST, martes a viernes)
- Datos vía Polygon / Massive API
- Totalmente funcional para análisis individual, escaneo multi-activo y alertas

## Instalación y uso local

```bash
git clone https://github.com/lfov256/squeeze-wave-tradingview.git
cd squeeze-wave-tradingview
pip install -r requirements.txt
streamlit run squeeze_tradingview.py
```

Requiere API key de Polygon (almacenada como secret `POLYGON_API_KEY` o variable de entorno). La app funciona tanto localmente como desplegada en Streamlit Cloud.

## Conceptos matemáticos clave

### Lambda Ω (ciclo dominante)
Longitud de onda dominante del activo, estimada mediante **análisis espectral de Welch** sobre precios suavizados (EMA). Permite adaptar automáticamente las ventanas de cálculo según el ritmo natural de cada mercado (el oro oscila más lento que el Bitcoin, por ejemplo).

### Squeeze (compresión)
Se detecta cuando las Bandas de Bollinger entran completamente dentro del Canal de Keltner:
- `UpperBB ≤ UpperKC` y `LowerBB ≥ LowerKC`
- Duración mínima: **3 días consecutivos** = episodio válido

### SqueezeIndex (0-100)
Mide la tensión acumulada combinando:
- Percentil histórico del ancho de BB (cuánto se ha estrechado *para ese activo*)
- Factor de calidad Lambda (penaliza ciclos irregulares o ruido)

Valores > 80 indican compresión extrema; 50-80 moderada.

### Trend compuesto
Dirección de la presión acumulada durante el squeeze, calculada con 4 componentes ponderados:
- Pendiente larga normalizada por ATR
- Pendiente corta
- Flujo de dinero (MFI)
- Velocidad (ROC)

La señal fuerte solo se activa cuando hay squeeze + Trend supera umbral configurable + filtro de volatilidad (evita señales cuando el mercado ya está explotando).

## Backtest por episodios
El backtest no usa retornos raw. Opera al **final de cada episodio** de squeeze:
1. Registra la dirección que predecía el modelo.
2. Mide el retorno real en los N días siguientes.
3. Alinea el retorno según la señal (long si alcista, short si bajista).

Métricas reportadas:
- Win Rate
- Expectancy por operación
- Profit Factor
- Sharpe aproximado (anualizado)
- Max Drawdown
- Calmar Ratio
- Payoff Ratio
- MFE / MAE por episodio

**Importante**: los retornos son brutos (sin comisiones, slippage ni gestión de posición). Sirven para evaluar si el modelo aporta información predictiva real, no para proyectar beneficios.

## Limitaciones (documentadas de forma objetiva)
- No predice el timing exacto de la ruptura (puede ocurrir en 1 día o en 2-3 semanas).
- Backtest in-sample; para validación robusta se recomienda reservar datos out-of-sample.
- Sin stops, targets ni position sizing (eso queda fuera del modelo).
- Edge modesto pero medible en los episodios detectados.
- Sample size limitado con ventanas de histórico cortas (recomendado ≥ 2 años para conclusiones más sólidas).

El modelo es una herramienta de **contexto y scanner de atención**, no un sistema de trading completo por sí solo.

## Componentes del repositorio

| Archivo / Carpeta              | Propósito                                      |
|--------------------------------|--------------------------------------------------|
| `squeeze_tradingview.py`      | App Streamlit completa (dashboard, backtest, escaneo multi-activo, email) |
| `daily_alerts.py`             | Lógica de generación y envío de alertas diarias     |
| `subscriptions.json`          | Lista de activos monitorizados para alertas      |
| `.github/workflows/`          | Automatización de alertas (GitHub Actions)     |
| `scripts/`                    | Utilidades (backtest standalone, descarga histórica) |
| `data/historical/`            | Datos cacheados en formato Parquet               |

## Alertas diarias
Las alertas se envían automáticamente a las 8:00 AM CEST (martes-viernes) con los activos que presentan compresión activa. Para modificar la lista de activos edita `subscriptions.json` o usa la interfaz de la app Streamlit.

Para forzar una alerta de prueba: ve a la pestaña Actions → workflow "Daily Squeeze Wave Email Alerts" → Run workflow.

## Filosofía del proyecto
Enfoque cuantitativo y pragmático: medir lo que realmente funciona con backtest transparente, documentar limitaciones sin adornos y mantener el sistema simple y operable. El objetivo no es predecir el futuro con precisión, sino identificar regímenes de alta probabilidad de movimiento significativo y actuar en consecuencia con disciplina.

---

**SqueezeIndex v3.0** — Compresión de ondas + análisis espectral + backtest riguroso