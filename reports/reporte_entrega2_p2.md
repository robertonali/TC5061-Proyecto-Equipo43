# Reporte de experimentos con MLflow, sección de P2

Borrador de la sección de P2 para `TC5061_Reporte_MLFlow_Equipo43.pdf`. Las cifras salen de los
runs del experimento `mibici-equipo43`, lanzados con `mlflow run` desde el commit `b6edd95`; las
decisiones completas están en `references/decisiones_p2.md` (decisiones 8, 9 y 10).

## Pregunta y métrica de negocio

**P2:** ¿una estación estará en riesgo de quedarse sin bicicletas ("se vacía") o sin anclajes
("se llena") en una ventana de 2 horas? Cada día a las 4:00 se predicen las 12 ventanas del día
para que el equipo de rebalanceo decida a qué estaciones ir y si debe llevar o retirar bicicletas.
La etiqueta se calcula con los viajes de los datos abiertos de MiBici (MiBici, s.f.), del
2022-03-01 al 2026-09-30.

La métrica para elegir el modelo es el promedio de la PR-AUC de "se vacía" y "se llena" en
validación (`val_pr_auc_promedio`). La PR-AUC es más informativa que la ROC-AUC cuando la clase de
interés es minoritaria (Saito & Rehmsmeier, 2015), y aquí las clases de riesgo son el 16% de las
ventanas de validación. Como el ML Canvas prioriza no dejar a una persona usuaria sin bicicleta o
sin anclaje, también se revisa el recall de cada clase de riesgo.

## Reproducción

Repositorio: https://github.com/robertonali/TC5061-Proyecto-Equipo43

El repositorio es un MLflow Project (MLflow, s.f.-b): `MLproject` declara el entorno y los
puntos de entrada `ingest`, `features` y `train`, y cada corrida queda registrada en MLflow
Tracking (MLflow, s.f.-c) con sus parámetros, métricas, etiquetas y artefactos. Con el servidor
de MLflow levantado (`mlflow server --backend-store-uri sqlite:///mlflow.db
--artifacts-destination mlruns --port 5000 --workers 1`) y `MLFLOW_TRACKING_URI` apuntando a él:

```powershell
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo43
mlflow run . -e features --env-manager local --experiment-name mibici-equipo43 -P question=p2
mlflow run . -e train --env-manager local --experiment-name mibici-equipo43 -P question=p2 -P model=frecuencia
mlflow run . -e train --env-manager local --experiment-name mibici-equipo43 -P question=p2 -P model=busqueda-hgb
mlflow run . -e train --env-manager local --experiment-name mibici-equipo43 -P question=p2 -P model=hgb -P learning_rate=0.05
python -m mibici_mlops.modeling.registrar_p2 --champion <run_id> --challenger <run_id>
```

El README tiene la lista completa (los dos baselines y los cuatro candidatos) y el resultado
esperado de cada paso.

## Tabla comparativa

Misma partición temporal que la entrega 1: entrenamiento hasta 2025-09-30 (5,247,312 ventanas),
validación de 2025-10-01 a 2026-03-31 (809,388) y prueba de 2026-04-01 a 2026-09-30 (922,584).
Mejora: cambio relativo de la PR-AUC promedio frente al baseline de frecuencia.

| Run | Modelo | Hiperparámetros clave | PR-AUC prom. validación | PR-AUC prom. prueba | PR-AUC se vacía / se llena (prueba) | Recall se vacía / se llena (prueba) | Mejora en prueba |
|---|---|---|---|---|---|---|---|
| Baseline: clase mayoritaria | `DummyClassifier` | `most_frequent` | 0.080 | 0.064 | 0.057 / 0.072 | 0 / 0 | −86% |
| **Baseline: frecuencia 4 semanas** | Regla | 4 semanas, mismo día y hora | 0.495 | 0.453 | 0.368 / 0.539 | 0.47 / 0.62 | — |
| Logística | `LogisticRegression` | `C` 1, `balanced` | 0.555 | 0.505 | 0.418 / 0.593 | 0.67 / 0.77 | +11% |
| HGB | `HistGradientBoostingClassifier` | lr 0.1, 31 hojas, 300 árboles, `balanced` | 0.570 | 0.521 | 0.441 / 0.601 | 0.71 / 0.79 | +15% |
| HGB sin pesos | `HistGradientBoostingClassifier` | lr 0.1, 31 hojas, 300 árboles, sin pesos | 0.573 | 0.523 | 0.443 / 0.603 | 0.26 / 0.45 | +15% |
| Búsqueda (6 nested runs) | `HistGradientBoostingClassifier` | lr 0.05 o 0.1 × 31, 63 o 127 hojas, `balanced` | 0.569 a 0.571 | — | — | — | — |
| **HGB, mejor de la búsqueda (champion)** | `HistGradientBoostingClassifier` | lr 0.05, 31 hojas, 300 árboles, `balanced` | **0.571** | **0.522** | **0.442 / 0.602** | **0.71 / 0.79** | **+15%** |

Costo: la logística entrena en 30 s y predice validación y prueba (1.7 millones de ventanas) en
0.4 s; los HGB, en 69 a 81 s y 8 a 10 s. Estabilidad: la desviación estándar mensual de la PR-AUC
promedio en prueba es de 0.033 en el champion y de 0.030 en el baseline.

**Cambios respecto a la entrega 1.** La partición y las métricas de prueba no cambian; las del
baseline de frecuencia son idénticas a las de la entrega 1 (PR-AUC de 0.368 y 0.539), así que la
comparación es válida. Se agregaron las mismas métricas en validación, porque ahí se elige el
modelo sin tocar la prueba; el promedio de las dos PR-AUC, para elegir con un solo número; su
variación mensual (estabilidad) y los tiempos de entrenamiento y predicción (costo). Los baselines
se volvieron a correr con `mlflow run` para tener esas métricas y las etiquetas nuevas.

## Análisis y modelo elegido

**Todos los candidatos superan al baseline.** Los candidatos son modelos de scikit-learn
(Pedregosa et al., 2011). La regresión logística mejora la PR-AUC promedio de
prueba un 11%, y los modelos de gradient boosting, un 15%. La diferencia entre la logística y HGB
indica que hay combinaciones de variables (por ejemplo, la hora pico de un día hábil en una
estación que suele vaciarse) que un modelo lineal no captura (Ke et al., 2017).

**La búsqueda casi no cambia el resultado.** Las 6 combinaciones de la rejilla quedan entre 0.569
y 0.571 en validación, y dar más hojas a cada árbol no ayuda. El límite está en la información de
las variables, no en la capacidad del modelo: la siguiente mejora tendría que venir de variables
nuevas (riesgo del día anterior, la demanda que pronostica P1, clima), no de más hiperparámetros.

**Los pesos de clase deciden el punto de operación, no la calidad del orden.** HGB con y sin pesos
tienen la misma PR-AUC (0.571 y 0.573), es decir, ordenan las ventanas por riesgo igual de bien.
Lo que cambia es la clase que predicen: sin pesos, el modelo casi siempre predice "sin riesgo" y
solo detecta el 26% de las ventanas que se vacían en prueba; con pesos balanceados detecta el 71%.
Compensar el desbalance con pesos de clase es una práctica reconocida (Lakshmanan et al., 2020).

**Champion: HGB con pesos de clase, `learning_rate` 0.05 y 31 hojas** (versión 1 de
`mibici-equipo43-p2-riesgo`):

- **Mejora sobre el baseline:** +15% de PR-AUC promedio en prueba (0.522 contra 0.453); detecta el
  71% de las ventanas que se vacían (47% el baseline) y el 79% de las que se llenan (62%).
- **Métrica de negocio:** es el de mayor recall en las dos clases de riesgo. Frente a HGB sin
  pesos, renuncia a 0.002 de PR-AUC, una diferencia del tamaño de la que hay entre combinaciones de
  la búsqueda.
- **Costo de ese recall:** su precisión en "se vacía" es de 0.30 (0.46 el baseline): unas 7 de
  cada 10 alertas de vaciado no se cumplen. Por eso la lista diaria se ordena por la probabilidad
  del modelo y el equipo atiende primero las alertas más altas.
- **Estabilidad:** varía de un mes a otro casi lo mismo que el baseline. Todos los modelos bajan de
  0.04 a 0.05 de validación a prueba porque la prueba tiene menos riesgo (13% de las ventanas
  contra 16%), no por sobreajuste.
- **Costo:** entrena en 81 s y la predicción de un día (471 estaciones × 12 ventanas) tarda menos
  de un segundo; el costo no separa a los candidatos.

**Challenger: regresión logística** (versión 2). Tiene 0.016 menos de PR-AUC en validación, pero
predice 20 veces más rápido, sus coeficientes se pueden explicar y es de otra familia: es una
alternativa real si el champion falla o si hay que justificar una alerta. El HGB con
`learning_rate` 0.1 sería una copia casi idéntica del champion.

## Model Registry y ciclo de vida

- **Nombre:** `mibici-equipo43-p2-riesgo`, con una descripción de la entrada, la salida y el
  esquema de aliases.
- **Linaje de cada versión** (etiquetas): experimento, run de origen, commit, si había cambios sin
  commit, versión del manifiesto de los datos (`c311be64…`), versión de DVC, familia del modelo y
  PR-AUC de validación y prueba.
- **Esquema: aliases.** `champion` marca la versión en producción y `challenger`, la siguiente
  opción. Se eligieron aliases y no etapas porque las etapas están obsoletas desde MLflow 2.9 y un
  alias se mueve de una versión a otra sin crear versiones nuevas ni cambiar el código que carga
  el modelo: promover o regresar a la versión anterior es mover el alias (MLflow, s.f.-a). Así, el
  registro funciona como el punto de control entre experimentación y producción que describen
  Treveil et al. (2020).
- **Verificación:** `models:/mibici-equipo43-p2-riesgo@champion` se carga con `mlflow.pyfunc`
  (clase predicha) y con `mlflow.sklearn` (probabilidad de cada clase).

## Capturas de P2

| Archivo sugerido | Qué mostrar |
|---|---|
| `reports/figures/mlflow2_p2_1_nested_runs.png` | Experimento `mibici-equipo43` filtrado con `tags.question = 'P2'`, con el run padre `p2-busqueda-hgb-balanceado-arboles300` expandido (sus 6 hijos) y las columnas `stage_of_work`, `val_pr_auc_promedio` y `test_pr_auc_promedio` |
| `reports/figures/mlflow2_p2_2_comparacion.png` | Los 2 baselines y los 4 candidatos (filtro `tags.question = 'P2' and tags.stage_of_work != 'search' and tags.stage_of_work != 'features'`) con las columnas `val_pr_auc_promedio`, `test_pr_auc_promedio` y el recall de prueba de cada clase de riesgo |
| `reports/figures/mlflow2_p2_3_artefactos.png` | Pestaña *Artifacts* del champion con `plots/`, `data/` y `config/` abiertas |
| `reports/figures/mlflow2_p2_4_modelo_firma.png` | El modelo que guardó el champion (*Logged models*) con su firma y ejemplo de entrada |
| `reports/figures/mlflow2_p2_5_registry.png` | *Models* → `mibici-equipo43-p2-riesgo` con los aliases `champion` y `challenger`, y el detalle de la versión 1 con sus etiquetas de linaje |

## Referencias

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM:
A highly efficient gradient boosting decision tree. En I. Guyon, U. V. Luxburg, S. Bengio,
H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in neural information
processing systems 30* (pp. 3146–3154). Curran Associates.

Lakshmanan, V., Robinson, S., & Munn, M. (2020). *Machine learning design patterns: Solutions to
common challenges in data preparation, model building, and MLOps*. O'Reilly Media.

MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. https://mibici.net/es/datos-abiertos/

MLflow. (s.f.-a). *MLflow Model Registry* [Documentación]. MLflow.

MLflow. (s.f.-b). *MLflow Projects* [Documentación]. MLflow.

MLflow. (s.f.-c). *MLflow Tracking* [Documentación]. MLflow.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M.,
Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M.,
Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
Learning Research, 12*, 2825–2830.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC
plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), Artículo
e0118432. https://doi.org/10.1371/journal.pone.0118432

Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A.,
Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the
enterprise*. O'Reilly Media.
