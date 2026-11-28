# Model Card: modelos del equipo 43 (MiBici MLOps)

Fichas de los tres modelos que sirve la API del proyecto, uno por pregunta de negocio. Siguen el
formato de *Model Cards for Model Reporting* (Mitchell et al., 2019), adaptado al proyecto: cada
ficha dice para qué sirve el modelo, con qué datos se entrenó, qué tan bien funciona, dónde puede
fallar y cómo se vigila.

| Pregunta | Modelo en el Model Registry | En producción (`champion`) | Responsables |
|---|---|---|---|
| [P1](#p1-demanda-de-viajes-por-estación-y-día): viajes por estación y día | `mibici-equipo43-p1-demanda` | `HistGradientBoostingRegressor` con pérdida de Poisson | Manuel Medina |
| [P2](#p2-riesgo-de-que-una-estación-se-vacíe-o-se-llene): riesgo de vaciado o llenado | `mibici-equipo43-p2-riesgo` | `HistGradientBoostingClassifier` con pesos de clase | Roberto |
| [P3](#p3-grupos-de-estaciones-con-patrones-de-uso-parecidos): grupos de estaciones | `mibici-equipo43-p3-clusters` | KMeans con 3 grupos sobre el perfil de uso | Iñaki Medina y Fernando Mejía |

## Cómo leer estas fichas

- **Producción es el alias `champion`.** No hay un alias `production` aparte: promover una versión o
  volver a la anterior es mover el alias (decisión 9 de P2).
- **Las versiones y los `run_id` dependen del registry.** Cada integrante tiene su propio servidor
  de MLflow, y `mlflow run . -e pipeline` crea un registry nuevo desde cero. Lo que identifica al
  modelo en cualquier registry es el nombre del run de origen. La API dice qué versión y qué run
  sirve en `GET /salud` y en el endpoint `modelo` de cada pregunta.
- **Mantenimiento:** la ficha de un modelo se actualiza en el mismo pull request que mueve su
  `champion` o cambia sus datos de entrenamiento.

## Datos comunes

Los tres modelos usan los [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), el
sistema de bicicletas públicas de Guadalajara: los viajes mensuales y el catálogo de estaciones.
Los datos crudos se versionan con DVC (`data/raw.dvc`). Las fuentes, los problemas conocidos y cómo
se corrigen están en [references/fuentes_datos.md](references/fuentes_datos.md).

---

## P1: demanda de viajes por estación y día

> **[POR COMPLETAR – P1]** Llenar cada sección con la misma estructura que la ficha de P2. Lo que
> ya está escrito sale del README y es solo un punto de partida.

- **Responsable:** Manuel Medina.
- **Decisiones:** [references/decisiones_p1.md](references/decisiones_p1.md).

### 1. Detalles del modelo

- **Modelo registrado:** `mibici-equipo43-p1-demanda`.
- **En producción (`champion`):** `HistGradientBoostingRegressor` con pérdida de Poisson y 63 hojas,
  del run `p1-hgb-poisson-lr0.1-hojas63-arboles300`.
- **`challenger`:** `RandomForestRegressor`, del run `p1-rf-arboles100-prof16-hoja20`.
- [POR COMPLETAR – P1] Hiperparámetros, semilla y etiquetas de linaje de la versión.

### 2. Uso previsto

- Cada día a las 23:59, predecir cuántos viajes iniciarán en cada estación el día siguiente. Lo usa
  operación para repartir bicicletas y personal por zona y por día.
- [POR COMPLETAR – P1] Usos fuera de alcance.

### 3. Datos

- [POR COMPLETAR – P1] Periodo, partición por fechas y variables (`lag_7d`, `media_7d`, calendario
  y datos de la estación).

### 4. Métricas y resultados

- MAE y RMSE en viajes por estación y día, en validación y prueba. El modelo se elige con el MAE de
  validación.
- [POR COMPLETAR – P1] Tabla del `champion` y el `challenger` contra el baseline (`lag_7d`).

### 5. Limitaciones y riesgos

- [POR COMPLETAR – P1]

### 6. Monitoreo y mantenimiento

- [POR COMPLETAR – P1] Drift con Evidently (`reports/evidently/p1_*`) y cuándo reentrenar.

### 7. Cómo usarlo

- `POST /prediccion` y `GET /modelo` de la API (ver la sección "La API" del README).

---

## P2: riesgo de que una estación se vacíe o se llene

- **Responsable:** Roberto.
- **Fecha de la ficha:** 2026-11-27.
- **Decisiones:** [references/decisiones_p2.md](references/decisiones_p2.md). Los números de esta
  ficha salen de ahí y de [reports/evidently/p2_reporte.md](reports/evidently/p2_reporte.md).

### 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo43-p2-riesgo` |
| En producción | Versión 1, alias `champion`: `models:/mibici-equipo43-p2-riesgo@champion` |
| Algoritmo | `HistGradientBoostingClassifier` de scikit-learn: pesos de clase balanceados, `learning_rate` 0.05, 31 hojas, 300 árboles y semilla 43 |
| Run de origen | `p2-hgb-balanceado-lr0.05-hojas31-arboles300`, la mejor combinación de la búsqueda `p2-busqueda-hgb-balanceado-arboles300` |
| Linaje | Cada versión lleva como etiquetas el run de origen, el commit, si había cambios sin commit, la versión del manifiesto de datos, la versión de DVC de `data/raw`, la familia del modelo y la PR-AUC de validación y prueba |
| Otras versiones | Versión 2, alias `challenger`: regresión logística balanceada (`p2-logistica-balanceado-c1.0`), la alternativa si el `champion` falla o hay que explicar una alerta. Versión 3, alias `reentrenado`: el `champion` reentrenado con seis meses más de datos, que perdió el A/B (sección 6); solo existe en el registry donde se corrió el A/B |

### 2. Uso previsto

- **Para qué:** cada día a las 4:00, estimar para las 12 ventanas de 2 horas del día (00:00,
  02:00, …, 22:00) la probabilidad de que cada estación se quede sin bicicletas (`se_vacia`) o sin
  anclajes (`se_llena`).
- **Quién lo usa:** el equipo de rebalanceo, para armar la lista diaria de estaciones a visitar y
  decidir si lleva o retira bicicletas. La lista se ordena por probabilidad. Los cortes propuestos
  (decisión 5) son 70% o más para alerta roja (ir primero), de 40% a 70% para amarilla (ir si hay
  tiempo) y menos de 40% sin acción. Son una propuesta para el equipo, no parte del modelo.
- **Fuera de alcance:**
  - Decir cuántas bicicletas hay en una estación: el modelo da un riesgo, no un inventario.
  - Actualizar el riesgo durante el día: las variables solo usan datos hasta las 3:59.
  - Decidir de forma automática, sin una persona: es una herramienta para priorizar.
  - Usarlo en otra ciudad u otro sistema de bicicletas sin reentrenarlo y volver a validarlo.

### 3. Datos

- **Fuente:** los viajes del 2022-03-01 al 2026-09-30 y la capacidad de cada estación
  (`dpcapacity`, del catálogo de estaciones).
- **Datos personales:** no usa los datos de los usuarios (id, género ni año de nacimiento); solo
  cuenta salidas y llegadas por estación.
- **Etiqueta, "riesgo sin intervención"** (decisiones 4 y 5): MiBici no publica cuántas bicicletas
  hay en cada estación, así que el inventario se simula. Cada estación empieza el día (4:00) al 50%
  de su capacidad, y se le suman las llegadas y se le restan las salidas. Una ventana es
  `se_vacia` si el inventario llega a 0 o menos, `se_llena` si llega a la capacidad o más, y
  `sin_riesgo` si no pasa ninguna de las dos.
- **Clases:** 81.4% `sin_riesgo`, 9.8% `se_vacia` y 8.8% `se_llena`, en 6,979,284 ventanas de 471
  estaciones.
- **Partición por fechas** (decisión 6): la prueba es lo más reciente, como cuando el modelo se use
  de verdad.

  | Conjunto | Periodo | Ventanas | `se_vacia` | `se_llena` |
  |---|---|---|---|---|
  | Entrenamiento | 2022-03-01 a 2025-09-30 | 5,247,312 | 10.8% | 9.2% |
  | Validación | 2025-10-01 a 2026-03-31 | 809,388 | 7.8% | 8.3% |
  | Prueba | 2026-04-01 a 2026-09-30 | 922,584 | 5.7% | 7.2% |

- **Variables** (decisión 8), seis, todas conocidas a las 4:00: `hora`, `dia_semana`, `feriado`,
  `capacidad`, `frec_se_vacia` y `frec_se_llena`. Las dos últimas son la fracción de las 4 semanas
  anteriores en que la misma estación, el mismo día de la semana y la misma ventana se vació o se
  llenó.

### 4. Métricas y resultados

- **Métrica para elegir el modelo:** el promedio de la PR-AUC de `se_vacia` y `se_llena` en
  validación. La prueba solo se reporta y no interviene en la elección. Se usa la PR-AUC porque las
  clases de riesgo son minoritarias: un modelo que siempre dice "sin riesgo" acierta el 87% de las
  ventanas de prueba sin detectar ningún riesgo.

| Modelo | PR-AUC promedio, validación | PR-AUC promedio, prueba | Recall en prueba (`se_vacia` / `se_llena`) | Precisión en prueba (`se_vacia` / `se_llena`) |
|---|---|---|---|---|
| Baseline: frecuencia de las 4 semanas anteriores | 0.495 | 0.453 | 0.47 / 0.62 | 0.46 / 0.59 |
| `challenger`: regresión logística | 0.555 | 0.505 | 0.67 / 0.77 | 0.32 / 0.43 |
| **`champion`: HGB con pesos de clase** | **0.571** | **0.522** | **0.71 / 0.79** | 0.30 / 0.42 |

- **Contra el baseline:** en prueba, la PR-AUC promedio sube 15% (20% en `se_vacia` y 12% en
  `se_llena`).
- **Punto de operación:** se prioriza el recall, porque es peor dejar a un usuario sin bicicleta o
  sin anclaje que mandar un camión de más. El modelo detecta el 71% de las ventanas que se vacían y
  el 79% de las que se llenan. A cambio, de cada 10 alertas de vaciado, unas 7 no se cumplen.
- **Estabilidad:** la PR-AUC promedio varía de un mes a otro casi lo mismo que la del baseline
  (desviación estándar de 0.033 contra 0.030 en prueba).
- **Costo:** entrena en 81 segundos con 5.2 millones de ventanas, y la predicción de un día
  (unas 470 estaciones × 12 ventanas) tarda menos de un segundo.

### 5. Limitaciones y riesgos

- **La etiqueta es una simulación.** Dice que la estación se quedaría vacía o llena si nadie la
  rebalanceara, no que lo estuvo. En el 9.8% de las ventanas el inventario simulado baja de 0 y en
  el 8.8% pasa la capacidad, lo que en la realidad no puede pasar: ahí hubo rebalanceo o la estación
  no empezó al 50%. Un inventario inicial estimado para cada estación sería más realista; queda como
  posible mejora.
- **Muchas falsas alarmas.** Con una precisión de 0.30 en `se_vacia`, atender todas las alertas
  manda camiones de más. Por eso la lista se ordena por probabilidad y se atienden primero las más
  altas.
- **Estaciones nuevas.** Unas 100 estaciones abrieron de mayo a agosto de 2026 y ninguna estaba en
  el entrenamiento. Son el 12% de las ventanas de prueba y casi nunca están en riesgo (alrededor
  del 3%).
- **El riesgo baja con el tiempo.** Las clases de riesgo pasan del 20% de las ventanas de
  entrenamiento al 13% de las de prueba. Como la PR-AUC depende de esa proporción, las cifras de
  validación y de prueba no se comparan entre sí.
- **El riesgo se concentra en pocas estaciones.** La mitad de las estaciones pasa menos del 9% de
  sus ventanas en riesgo, y el 10% pasa más de un tercio. Ordenar por probabilidad tiende a mandar
  al equipo a las mismas estaciones; conviene revisar que las zonas con poco historial no queden
  desatendidas.
- **Variables limitadas.** No usa el clima, los eventos ni la demanda que pronostica P1. La búsqueda
  de hiperparámetros casi no cambió el resultado (de 0.569 a 0.571): el límite está en las
  variables, no en el modelo. Los feriados son los oficiales; un evento o un feriado local fuera de
  esa lista no se ve.

### 6. Monitoreo y mantenimiento

- **Drift y calidad de los datos con Evidently** (decisión 12): la referencia es el entrenamiento, y
  la validación y la prueba son los dos periodos actuales. Se vigilan 8 columnas: las 6 variables,
  la etiqueta y la clase que predice el `champion`. Hay drift si la distancia es de 0.1 o más
  (Wasserstein normalizada en las numéricas, Jensen-Shannon en las categóricas).
  - Solo `frec_se_vacia` pasa el umbral: 0.11 en validación y 0.21 en prueba. El riesgo real baja:
    las ventanas que se vacían pasan del 10.8% al 7.8% y al 5.7%.
  - Los seis chequeos de calidad (faltantes, duplicados, rangos, horas, capacidad y clases) dan 0.
  - La PR-AUC promedio baja de 0.571 a 0.522, pero el baseline baja casi lo mismo y la ventaja del
    `champion` se mantiene (+0.076 y +0.069).
  - Reportes: `reports/evidently/p2_drift_calidad_validacion.html`, `p2_drift_calidad_prueba.html`
    y la interpretación en [p2_reporte.md](reports/evidently/p2_reporte.md).
- **A/B fuera de línea** (decisión 13): la versión 3 (el `champion` reentrenado con entrenamiento y
  validación) contra la versión 1, sobre las mismas 922,584 ventanas de prueba, con un bootstrap por
  días de 500 réplicas. La versión 3 mejora la PR-AUC promedio en +0.0028, con intervalo de
  confianza de 95% de +0.0024 a +0.0032. Es menos que el margen de 0.005 fijado antes de correr el
  A/B, así que la versión 1 sigue como `champion`.
- **Cuándo reentrenar** (`p2_reporte.md`, sección 6): correr el monitoreo con cada mes nuevo de
  datos, y reentrenar y correr el A/B si pasa alguna de estas señales:
  - la ventaja del `champion` sobre el baseline baja de +0.05 dos meses seguidos;
  - el recall de `se_vacia` baja de 0.65 dos meses seguidos;
  - el drift de la clase predicha pasa 0.1 tres meses seguidos.

  El siguiente reentrenamiento recomendado es con los datos de octubre a diciembre de 2026, que ya
  incluyen a las estaciones nuevas. Como respaldo, una vez al año.
- **Promoción y vuelta atrás:** una versión nueva solo pasa a producción si cumple la regla del A/B:
  mejora la PR-AUC promedio en al menos 0.005, el intervalo de confianza queda arriba de 0 y el
  recall de ninguna clase de riesgo baja más de 0.01. `ab_p2 --promover` mueve `champion` y deja la
  versión anterior con el alias `anterior`. Volver atrás es mover el alias, volver a exportar el
  modelo y reconstruir la imagen de la API.
- **Comandos:** `make monitoreo_p2`, `make reentrenar_p2` y `make ab_p2`. Sin `make`:
  `python -m mibici_mlops.monitoreo p2`,
  `python -m mibici_mlops.modeling.train_p2 --modelo hgb --learning-rate 0.05 --reentrenamiento` y
  `python -m mibici_mlops.modeling.ab_p2`.

### 7. Cómo usarlo

- **API:** `POST /p2/prediccion` recibe hasta 10,000 filas por petición (una por estación y
  ventana) y regresa la clase y la probabilidad de cada clase. La clase es la más probable y, en
  empate, gana el riesgo. `GET /p2/modelo` dice qué versión y qué run se están sirviendo. Por
  ejemplo, un martes a las 8:00 en una estación que se vació en 2 de las 4 semanas anteriores:

  ```json
  {"filas": [{"estacion_id": 2, "hora": 8, "dia_semana": 1, "feriado": false, "capacidad": 23,
              "frec_se_vacia": 0.5, "frec_se_llena": 0.0}]}
  ```

  regresa `"clase": "se_vacia"`, con probabilidades de 0.873 (`se_vacia`), 0.005 (`se_llena`) y
  0.122 (`sin_riesgo`). El comando completo está en la sección "La API" del README.
- **Desde Python:** `mlflow.sklearn.load_model("models:/mibici-equipo43-p2-riesgo@champion")` y
  `predict_proba`. Se carga con `mlflow.sklearn` y no con `pyfunc`, porque `pyfunc` solo da la clase.

---

## P3: grupos de estaciones con patrones de uso parecidos

> **[POR COMPLETAR – P3]** Llenar cada sección con la misma estructura que la ficha de P2. Lo que
> ya está escrito sale del README y es solo un punto de partida.

- **Responsables:** Iñaki Medina y Fernando Mejía.
- **Decisiones:** [references/decisiones_p3.md](references/decisiones_p3.md).

### 1. Detalles del modelo

- **Modelo registrado:** `mibici-equipo43-p3-clusters`.
- **En producción (`champion`):** KMeans con 3 grupos sobre el perfil de uso, del run
  `p3-kmeans_k3`.
- **`challenger`:** `GaussianMixture` con 2 componentes, del run `p3-gmm_diag_k2`.
- **`reentrenado`:** la versión alternativa de la entrega 8, que se comparó con el `champion` en un
  A/B y no lo reemplazó.
- [POR COMPLETAR – P3] Hiperparámetros, semilla y etiquetas de linaje de la versión. Hoy las
  versiones de P3 guardan el commit y si había cambios sin commit, pero no `version_manifiesto` ni
  `version_datos`, que sí guardan P1 y P2.

### 2. Uso previsto

- Agrupar las estaciones por su patrón de uso. Lo usa planeación urbana para decidir dónde y de qué
  tipo abrir estaciones nuevas, y para planear por tipo de estación en lugar de estación por estación.
- [POR COMPLETAR – P3] Usos fuera de alcance.

### 3. Datos

- Viajes del 2022-03-01 al 2026-09-30 y ubicación de cada estación. No usa datos de los usuarios.
- Una fila por estación con su perfil semanal: la fracción de sus salidas y de sus llegadas en cada
  día de la semana y franja del día (56 variables). Perfil de entrenamiento de 2022-03 a 2025-09 y
  de prueba de 2026-04 a 2026-09.
- [POR COMPLETAR – P3]

### 4. Métricas y resultados

- Silhouette, Davies-Bouldin y Calinski-Harabasz sobre el perfil de uso, y estabilidad de los grupos
  entre periodos (`test_ari_estabilidad`).
- [POR COMPLETAR – P3] Tabla del `champion` y el `challenger` contra el baseline (zonas de la ciudad).

### 5. Limitaciones y riesgos

- [POR COMPLETAR – P3] Por ejemplo, que los números de grupo no se conservan entre versiones.

### 6. Monitoreo y mantenimiento

- [POR COMPLETAR – P3] Drift con Evidently, A/B (`ab_p3`) y cuándo reentrenar.

### 7. Cómo usarlo

- `POST /p3/prediccion` y `GET /p3/modelo` de la API (ver la sección "La API" del README).

---

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji,
  I. D. y Gebru, T. (2019). *Model Cards for Model Reporting*. Proceedings of the Conference on
  Fairness, Accountability, and Transparency (FAT\* '19). https://arxiv.org/abs/1810.03993
