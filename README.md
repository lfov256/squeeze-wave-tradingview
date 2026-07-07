# Squeeze Wave - Alertas por Email (Totalmente Operativo)

**Estado:** Todo está funcionando automáticamente.

- Alertas por email a las **8:00 AM CEST** (martes a viernes)
- Proveedor de datos: **Massive** (rebranding desde Polygon)
- La clave de Massive está guardada en el secret `POLYGON_API_KEY`

## Única acción que hiciste
Añadiste los 3 secrets en GitHub. Ya está.

## Cómo probar ahora (recomendado)

1. Ve a: https://github.com/lfov256/squeeze-wave-tradingview/actions
2. Busca el workflow **Daily Squeeze Wave Email Alerts**
3. Pulsa **Run workflow** → **Run workflow**
4. En 1-3 minutos recibirás un email de prueba.

## Si quieres cambiar los activos
Edita el archivo `subscriptions.json` directamente en GitHub (o pega el bloque de interfaz en la app Streamlit cuando quieras).

El sistema matemático de compresión de ondas + análisis espectral ya está completamente operacionalizado con Trade Republic.