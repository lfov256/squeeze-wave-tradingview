# Squeeze Wave TradingView

**SqueezeIndex v3.0** — Detección de compresión de volatilidad mediante física de ondas + análisis espectral (Welch) + backtest riguroso por episodios.

## Novedad: Sistema de Alertas Automatizadas Diarias

Ahora puedes suscribirte desde la interfaz a los activos del escaneo multi-activo y recibir alertas automáticas **todos los días a las 8:00 AM (martes a viernes)** cuando haya señal FUERTE.

### Cómo activar las alertas (3 pasos)

1. **Ejecuta la app Streamlit** (`streamlit run squeeze_tradingview.py`)
2. Ve a la pestaña **🔭 Escaneo Multi-Activo** y pulsa "Escanear todos los activos"
3. En la sección inferior **🔔 Suscripciones a Alertas Diarias**:
   - Selecciona los activos que quieres monitorear
   - Pulsa **"Suscribirme a TODOS los que aparecen"** o guarda selección manual
   - (Opcional) Ajusta los umbrales de SqueezeIndex y SignalStrength

Esto crea/actualiza el archivo `subscriptions.json` en la misma carpeta.

### Setup del script de alertas (una sola vez)

1. Crea un bot de Telegram:
   - Abre Telegram → busca @BotFather → `/newbot` → copia el token
   - Habla con tu bot una vez
   - Abre @userinfobot para obtener tu `chat_id`

2. Edita `daily_alerts.py` (o usa variables de entorno):
   ```bash
   export POLYGON_API_KEY="tu_key"
   export TELEGRAM_BOT_TOKEN="123456:ABC..."
   export TELEGRAM_CHAT_ID="tu_chat_id"
   ```

3. Prueba manualmente:
   ```bash
   python daily_alerts.py
   ```

4. Programa en cron (martes a viernes 8:00 AM):
   ```bash
   crontab -e
   ```
   Añade la línea:
   ```
   0 8 * * 2-5 /usr/bin/python3 /ruta/absoluta/a/daily_alerts.py >> /ruta/absoluta/alerts.log 2>&1
   ```

### Flujo operativo con Trade Republic

Cada mañana (Tue-Fri) a las 8:00 recibes en Telegram solo los activos suscritos que tienen **compresión de ondas fuerte + dirección clara** (SqueezeDetected + SI ≥ 75 + SignalStrength ≥ 0.60).

- Abre Trade Republic
- Revisa el gráfico en la app SqueezeIndex v3.0 (o TradingView)
- Aplica tu gestión de riesgo (riesgo 0.5-1% por trade, stop basado en ATR o invalidación del squeeze)
- Ejecuta manualmente

El edge matemático (compresión de energía de ondas + ciclo dominante Lambda) está ahora operacionalizado de forma diaria y limpia.

### Notas
- El script usa datos de cierre del día anterior (Polygon).
- No ejecuta órdenes automáticamente (Trade Republic no lo permite de forma segura).
- Nunca subas tokens al repo. Usa variables de entorno o `.env` + `python-dotenv`.
- Si quieres alertas también por email, el código ya tiene la función `send_email_resend` lista para extender.

## Instalación original

```bash
pip install -r requirements.txt
streamlit run squeeze_tradingview.py
```

## Licencia
Privado — lfov256
