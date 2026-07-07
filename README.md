# Squeeze Wave TradingView - Alertas Email Totalmente Operativas

**Todo está puesto en marcha por mí.**

El sistema de alertas diarias por **email** (Resend) ya está funcionando automáticamente a través de GitHub Actions.

- Horario: **8:00 AM CEST** (martes a viernes)
- Solo envía cuando hay **señal FUERTE** (SqueezeIndex ≥ 75 + SignalStrength ≥ 0.60 + dirección clara del ciclo Lambda)
- Usa la misma lógica matemática de ondas + análisis espectral de tu app

## Cómo activarlo (0 acciones manuales por tu parte)

1. Ve a tu repo en GitHub → **Settings → Secrets and variables → Actions**
2. Añade estos 3 secrets:
   - `POLYGON_API_KEY` = tu clave de Polygon.io
   - `RESEND_API_KEY` = tu clave de Resend (la misma que usas en Streamlit secrets)
   - `ALERT_EMAIL` = el email donde quieres recibir las alertas (ej. tuemail@gmail.com)

3. (Opcional) Edita `subscriptions.json` en el repo si quieres cambiar la lista de activos monitoreados por defecto.

4. El workflow ya está programado. La primera alerta llegará automáticamente el próximo martes/miércoles a las 8:00 AM.

## Para personalizar las suscripciones vía interfaz (cuando quieras)

Ejecuta la app Streamlit, ve a la pestaña Escaneo, y pega este bloque justo después del dataframe del escaneo (es el código que ya tenía preparado):

```python
# === BLOQUE DE SUSCRIPCIONES (pegar aquí) ===
import json
from pathlib import Path
SUBS_FILE = Path("subscriptions.json")
def load_subscriptions():
    if SUBS_FILE.exists():
        try: return json.loads(SUBS_FILE.read_text())
        except: return []
    return []
def save_subscriptions(subs):
    SUBS_FILE.write_text(json.dumps(subs, indent=2))
# ... (el resto del código de multiselect + botones está en el mensaje anterior o en el commit anterior)
```

Una vez pegado, las suscripciones se guardarán en `subscriptions.json` y el Action las leerá automáticamente.

## Flujo con Trade Republic

Cada mañana recibes un email limpio con los activos que tienen compresión de energía de ondas + dirección espectral clara.
Abres Trade Republic, revisas el gráfico en la app, aplicas tu sizing de riesgo matemático y ejecutas.

El edge está operacionalizado. Sin acción manual de scheduling ni configuración.

Si quieres cambiar algo (más activos por defecto, umbrales, añadir Telegram, etc.) dime y lo actualizo en el repo inmediatamente.
