# Decisiones de P1: demanda de viajes por estación y día

Decisiones específicas de P1: ¿cuántos viajes iniciarán en cada estación durante el día
siguiente? Las decisiones generales del proyecto (formato de los datos, periodo de análisis)
están en [decisiones.md](decisiones.md). El código está en `mibici_mlops/features.py` (subcomando
`p1`) y `mibici_mlops/modeling/train_p1.py`.

## 1. Predecir los viajes por estación y día, con horizonte de un día

- **Fecha:** 2026-10-18
- **Estado:** vigente
- **Aplica a:** la variable objetivo, las filas y las predicciones de P1.

### Contexto

La actividad pide una regresión sobre la demanda sin fijar la granularidad. La propuesta (Tabla 1)
fija estación × día con horizonte de un día, generado por lotes. El pipeline debe coincidir.

### Decisión

- **Variable objetivo:** número de viajes que inician en la estación durante el día (`viajes`),
  con la estación como `origen_id`.
- **Unidad:** una fila por estación y día. La malla es completa: los días sin viajes cuentan
  como 0.
- **Cuándo se predice:** una vez al día, al cierre del día anterior (23:59), para el día
  siguiente.

### Por qué

- **Coincide con la decisión del operador.** Reparte bicicletas y personal por zona y por día,
  con un pronóstico por lotes.
- **Es menos ruidosa y más ligera.** Hay unas 24 veces menos filas que por hora, y los conteos
  diarios tienen menos ceros.
- **La malla completa enseña también los días sin demanda.** Si solo se usaran los días con
  viajes, el modelo nunca vería un 0 y sobrestimaría la demanda.

### Alternativa descartada

**Estación × hora.** Dice a qué hora hace falta la flota, pero P2 ya cubre el detalle dentro del
día con ventanas de 2 horas, y la propuesta fija P1 por día.

### Consecuencias

- Hay una predicción por estación y día.
- Se pueden sumar las estaciones de una zona para dimensionar la flota por zona.

## 2. Variables de historial que terminan en el cierre del día anterior

- **Fecha:** 2026-10-18
- **Estado:** vigente
- **Aplica a:** todas las variables de P1 construidas con viajes pasados.

### Contexto

La corrida diaria predice el día D con datos hasta las 23:59 del día D-1. Una variable que use un
viaje del día D se calcularía bien al entrenar, pero no existiría al predecir (fuga de datos).

### Decisión

Cada variable de historial usa solo días anteriores a D:

| Variable | Qué es |
|---|---|
| `lag_7d` | Viajes del mismo día de la semana anterior (también es el baseline) |
| `media_7d` | Promedio de los 7 días anteriores (D-7 a D-1) |

Se agregan variables de calendario (`dia_semana`, `mes`, `es_fin_de_semana`, `es_feriado`) y de la
estación (`capacidad`, `lat`, `lon`). Son las de la Tabla 9 de la propuesta.

### Por qué

- **Se siguen las variables de la Tabla 9 de la propuesta.** `lag_7d` recoge la estacionalidad
  semanal y `media_7d` resume el nivel reciente de la estación.
- **Se comprueba con una prueba.** `tests/test_features_p1.py` altera los viajes desde el día D y
  verifica que ninguna variable de ese día cambie.

### Alternativas descartadas

- **Ventanas que incluyen el día D:** infla las métricas porque usa información que no existe al
  predecir.
- **Identificador de la estación como variable:** el gradient boosting de scikit-learn limita
  las variables categóricas a 255 valores, y la red puede tener más estaciones que eso. La
  ubicación y la capacidad ya identifican a cada estación.

### Consecuencias

- Los primeros 7 días de datos (desde el 1 de marzo de 2022) no tienen una semana completa de
  historial y quedan fuera: el entrenamiento efectivo empieza el 8 de marzo de 2022.
- Los días feriados son los de descanso obligatorio de la Ley Federal del Trabajo (art. 74).
  No incluyen eventos locales de la ciudad. **Pendiente:** revisar si algún evento local mueve la
  demanda lo suficiente para agregarlo.

## 3. Qué filas entran: estaciones con actividad en la semana previa

- **Fecha:** 2026-10-18
- **Estado:** vigente
- **Aplica a:** las filas de P1 en entrenamiento, validación y prueba.

### Contexto

Con la malla completa, una estación que aún no existía, o que ya cerró, tendría ceros que no son
demanda baja sino ausencia de servicio. Quitarlas con la fecha del primer o último viaje usaría
información del futuro.

### Decisión

Una fila (estación, día) se conserva solo si la estación tuvo al menos un viaje en los 7 días
anteriores al corte. Es una regla que se puede evaluar al predecir.

### Por qué

- **Evita ceros falsos** en estaciones nuevas o cerradas.
- **Es consistente con la propuesta:** las estaciones sin historial se reportan aparte.

### Consecuencias

- Una estación nueva entra a los datos desde el día siguiente a su primer viaje; sus rezagos de
  antes de abrir valen 0. Es poco frecuente en una red estable, pero conviene vigilarlo.
- En la prueba, las estaciones sin historial de entrenamiento se reportan en el análisis de error
  por nivel de uso como "sin historial".
- La malla incluye todas las estaciones de la nomenclatura (`dpcapacity`, latitud y longitud); los
  viajes de estaciones que no están ahí se descartan y se avisa cuántos.

## 4. Partición temporal y evaluación

- **Fecha:** 2026-10-18
- **Estado:** reemplazada por la decisión 6 (2026-10-19)
- **Aplica a:** `conjunto` en `features_p1.py` y `train_p1.py`.

### Decisión

| Conjunto | Periodo |
|---|---|
| Entrenamiento | 2022-03-08 a 2025-12-31 |
| Validación | 2026-01-01 a 2026-04-30 |
| Prueba | 2026-05-01 a 2026-09-30 |

- Los hiperparámetros se eligen con la validación.
- La prueba se evalúa **una sola vez**, con el modelo ya elegido.
- Las fechas de corte salen de `config.py` si existen `FECHA_FIN_ENTRENAMIENTO` y
  `FECHA_FIN_VALIDACION`; si no, se usan los valores de arriba.

### Consecuencias

- **Pendiente:** septiembre de 2026 está en la prueba y tiene el nivel de uso más alto de la serie
  (decisión 2 de `decisiones.md`). Falta decidir cómo tratarlo.
- **Pendiente:** acordar con el equipo que las fechas vayan en `config.py` para los tres modelos.

## 5. Baseline y modelo candidato

- **Fecha:** 2026-10-18
- **Estado:** vigente
- **Aplica a:** `train_p1.py`.

### Decisión

- **Baseline:** predecir la demanda del mismo día de la semana pasada (`lag_7d`).
- **Candidato:** `HistGradientBoostingRegressor`, sin parada anticipada (`early_stopping=False`).
  Se comparan tres configuraciones con el MAE de validación: dos con pérdida de Poisson (31 y 63
  hojas máximas) y una con error cuadrático. Se queda la de menor MAE.
- **Métricas:** `test_mae` y `test_rmse`, en viajes por estación-día; también `val_mae` y
  `val_rmse`. Las predicciones se acotan en 0.
- **Análisis de error:** por día de la semana y por nivel de uso de la estación (terciles de la
  demanda media en entrenamiento), para ver si el error es peor en estaciones de poco uso.

### Por qué

- **El baseline es una regla simple que el operador podría aplicar a mano.** Si el modelo no la
  supera, no vale la pena.
- **La demanda es un conteo no negativo, con muchos ceros**, por lo que se prueba la pérdida de
  Poisson además del error cuadrático.
- **MAE y RMSE** se leen en viajes: el MAE dice cuánto se equivoca la predicción típica y el
  RMSE castiga más los errores grandes de los días de mayor demanda. No se usa MAPE porque hay estación-días con 0
  viajes.
- **Sin parada anticipada:** la de scikit-learn aparta filas al azar del entrenamiento, lo que
  mezcla fechas; es preferible fijar el número de iteraciones y elegir con la validación.

### Consecuencias

- Cada corrida se registra en MLflow (experimento `mibici-equipo43`, etiqueta `pregunta=P1`,
  runs `p1-baseline-semana-previa` y `p1-gradient-boosting`).
- Los resultados con datos reales se anotan en el documento de la propuesta, no aquí.
- **Pendiente:** el tratamiento de septiembre de 2026 en la prueba (decisión 6).

## 6. Usar la partición temporal común del equipo

- **Fecha:** 2026-10-19
- **Estado:** vigente; reemplaza a la decisión 4
- **Aplica a:** `asignar_conjunto` en `features.py` (subcomando `p1`) y `train_p1.py`.

### Contexto

La decisión 4 proponía fechas propias para P1. Mientras tanto, el equipo fijó en `config.py` una
partición común para las tres preguntas (`FECHA_INICIO_VALIDACION` y `FECHA_INICIO_PRUEBA`,
decisión 6 de `decisiones_p2.md`). Con fechas distintas, los resultados de P1 no se podían
comparar con los de P2 y P3.

### Decisión

Usar las fechas de `config.py`:

| Conjunto | Periodo | Filas estación-día |
|---|---|---|
| Entrenamiento | 2022-03-08 a 2025-09-30 | 434,345 |
| Validación | 2025-10-01 a 2026-03-31 | 66,890 |
| Prueba | 2026-04-01 a 2026-09-30 | 75,913 |

El entrenamiento empieza el 2022-03-08 y no el 2022-03-01 (`FECHA_INICIO`) porque cada fila
necesita una semana completa de historial (decisión 2). Siguen igual las reglas de la decisión 4:
los hiperparámetros se eligen con la validación y la prueba se evalúa una sola vez.

### Por qué

- **Mismos días de prueba para P1, P2 y P3.** Los resultados de las tres preguntas se pueden
  comparar y reportar juntos.
- **Las fechas se definen una sola vez.** P1 ya no tiene valores propios de respaldo; si el equipo
  cambia la partición en `config.py`, P1 la sigue sin tocar su código.
- **Validación y prueba duran seis meses cada una**, en lugar de cuatro y cinco: la validación
  cubre otoño e invierno y la prueba, primavera y verano.

### Consecuencias

- Cambian los números de P1 (antes, `test_mae` de 5.97 para el gradient boosting y 7.85 para el
  baseline). Con esta partición, el gradient boosting tiene `val_mae` 5.72 y `test_mae` 5.99, y el
  baseline 7.78 y 7.94. La mejora sigue siendo de ~25%.
- Las variables de P1 no cambian: solo cambia la columna `conjunto`.
- La prueba empieza un mes antes e incluye septiembre de 2026 (pendiente de la decisión 2 de
  `decisiones.md`).
