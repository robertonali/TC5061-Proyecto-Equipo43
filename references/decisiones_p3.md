# Decisiones de P3: grupos de estaciones con patrones de uso parecidos

Decisiones específicas de P3: ¿qué grupos de estaciones comparten patrones de uso y dónde se
ubican en la ciudad? Las decisiones generales del proyecto (formato de los datos, periodo de
análisis) están en [decisiones.md](decisiones.md). La exploración que las respalda está en
`notebooks/1.0-im-initial-eda.ipynb`, y el código en `mibici_mlops/features.py` (subcomando
`p3`) y `mibici_mlops/modeling/train_p3.py`.

## 1. Agrupar estaciones por su perfil semanal de uso

- **Fecha:** 2026-10-18
- **Estado:** vigente
- **Aplica a:** P3.

### Contexto

P3 es la única pregunta sin variable objetivo: no hay una "respuesta correcta" de a qué grupo
pertenece cada estación. Había que decidir qué se agrupa, con qué granularidad y para qué
periodo.

### Decisión

| Aspecto | Decisión |
|---|---|
| Tarea | Clustering (no supervisado) |
| Qué se agrupa | Estaciones del catálogo `nomenclatura_2026_09.csv` |
| Granularidad | Una fila por estación, descrita por su perfil de uso semanal (decisión 2) |
| Horizonte | Los grupos se aprenden con el periodo de entrenamiento y se revisa que se mantengan en los 6 meses de prueba (decisión 4) |
| Quién lo usa | Planeación urbana y expansión del sistema |
| Decisión que toma | Dónde y de qué tipo abrir estaciones nuevas, y qué estaciones existentes se parecen para planear capacidad y rebalanceo por grupo, no estación por estación |

### Por qué

- **Planeación urbana piensa en tipos de estación**, no en 470 estaciones sueltas: "estaciones
  de oficinas", "de vivienda", "recreativas". Un grupo resume cómo se usa una zona.
- **El "dónde" sale del mapa de los grupos.** Si estaciones del mismo grupo quedan juntas, el
  grupo describe una zona; si quedan dispersas, el patrón depende del tipo de lugar (por
  ejemplo, junto al tren ligero) más que del barrio.

## 2. Variables: fracción de salidas y llegadas por día de la semana y franja del día

- **Fecha:** 2026-10-18
- **Estado:** vigente

### Decisión

Cada estación se describe con **56 fracciones**: de todas sus salidas, qué fracción ocurre en
cada combinación de día de la semana (7) y franja del día (4); y lo mismo con sus llegadas. Cada
grupo de 28 suma 1. Las franjas, definidas en `config.P3_FRANJAS`, son madrugada (0–6 h),
mañana (6–12 h), tarde (12–18 h) y noche (18–24 h), las mismas de la exploración del notebook.

### Por qué

- **Fracciones, no conteos.** Con conteos, los grupos solo separarían estaciones muy usadas de
  poco usadas; con fracciones se comparan patrones. El volumen (`salidas_dia`, `llegadas_dia`)
  se guarda aparte para describir los grupos, no para formarlos.
- **Salidas y llegadas juntas.** Una estación de vivienda tiene salidas en la mañana y
  llegadas en la tarde; una de trabajo, al revés. Con solo las salidas (como en el notebook) las
  dos se confunden si tienen el mismo total por franja.
- **Cuatro franjas.** Bastan para separar los usos que importan (ir al trabajo o a la escuela,
  regresar, salir de noche) sin repartir los viajes en tantas columnas que el perfil se llene
  de ruido. La hora de inicio de cada franja se cambia en un solo lugar.

### Datos personales

No se usa `usuario_id`, `genero` ni `anio_nacimiento`: solo la hora y las estaciones de cada
viaje. Los artefactos de MLflow tienen solo datos públicos del catálogo (id, nombre, latitud y
longitud de cada estación) y agregados por estación.

## 3. Mínimo de 500 salidas y 500 llegadas por periodo

- **Fecha:** 2026-10-18
- **Estado:** vigente

Una estación entra al agrupamiento de un periodo solo si tiene al menos `config.P3_MIN_VIAJES`
salidas y llegadas en él. Con pocos viajes las 28 fracciones son casi todas 0 o 1 y la estación
cae en un grupo por azar. 500 viajes en el periodo de entrenamiento (~1,300 días) es menos de un
viaje cada dos días: solo deja fuera estaciones que casi no se usan o que abrieron hace poco.

Con los datos actuales no excluye ninguna estación de entrenamiento: las 371 que tienen viajes
antes de octubre de 2025 superan el mínimo (la que menos tiene, 1,143 salidas). Las 113
estaciones restantes del catálogo no tienen ni un viaje en ese periodo porque abrieron después;
64 de ellas tienen perfil en prueba y el modelo les asigna un grupo.

## 4. Partición temporal: grupos con entrenamiento, estabilidad con prueba

- **Fecha:** 2026-10-18
- **Estado:** vigente

### Decisión

Se usan las mismas fechas que P2 (`config.py`): el perfil de **entrenamiento** usa los viajes
del 2022-03-01 al 2025-09-30, y el de **prueba**, del 2026-04-01 al 2026-09-30. El periodo de
validación (2025-10 a 2026-03) no se usa porque P3 no ajusta hiperparámetros con él.

### Por qué

- **Sin información del futuro.** Los grupos y el número de grupos se eligen solo con el
  periodo de entrenamiento. El de prueba se usa para medir, nunca para ajustar.
- **Un agrupamiento útil para planear debe durar.** Si una estación cambia de grupo de un año
  a otro, el grupo no sirve para decidir dónde abrir estaciones. Por eso se mide la estabilidad
  (decisión 6) asignando el perfil de prueba a los grupos aprendidos.

## 5. Número de grupos: 3

- **Fecha:** 2026-10-18
- **Estado:** vigente

Se probó KMeans con 2 a 8 grupos (`config.P3_GRUPOS_BUSQUEDA`) y se eligió el número con
mejor silhouette en entrenamiento, que coincide con lo que encontró la exploración del notebook
(3 grupos): 0.303, contra 0.277 con 2 grupos y 0.209 con 4. La curva se registra en MLflow
como `silhouette_por_numero_de_grupos.png` y la métrica `train_silhouette_busqueda`, con un
paso por número de grupos. Con 3 grupos, además, los colores del mapa se distinguen con
daltonismo.

## 6. Baseline, modelo candidato y métricas

- **Fecha:** 2026-10-18
- **Estado:** vigente

### Decisión

| | Modelo | Variables |
|---|---|---|
| Baseline | KMeans con 3 grupos | Latitud y longitud estandarizadas: zonas de la ciudad |
| Candidato | KMeans con 3 grupos | Las 56 fracciones del perfil de uso, sin estandarizar |

Los dos se evalúan **en el espacio del perfil de uso**:

| Métrica | Qué mide | Mejor |
|---|---|---|
| `train_silhouette`, `test_silhouette` | Qué tan cerca está cada estación de su grupo comparado con el siguiente más cercano | Más alto (máx. 1) |
| `train_davies_bouldin`, `test_davies_bouldin` | Separación de los grupos respecto a su dispersión | Más bajo |
| `train_calinski_harabasz`, `test_calinski_harabasz` | Varianza entre grupos contra varianza dentro de ellos | Más alto |
| `test_ari_estabilidad` | Qué tanto conserva cada estación su grupo entre entrenamiento y prueba (Adjusted Rand Index) | Más alto (máx. 1) |

### Por qué

- **El baseline responde "¿hace falta ver el uso?".** Agrupar por cercanía es lo que haría
  alguien con solo el mapa. Si el perfil de uso no da grupos más coherentes que las zonas, no
  aporta nada a planeación urbana.
- **Evaluar los dos en el mismo espacio** permite compararlos: el baseline siempre ganaría si
  se evaluara en latitud y longitud.
- **Silhouette, Davies-Bouldin y Calinski-Harabasz** son las métricas internas de la guía de
  clustering de scikit-learn: no necesitan etiquetas reales, que P3 no tiene. Se registran las
  tres porque cada una favorece formas de grupo distintas.
- **ARI de estabilidad.** Para el baseline vale 1 por construcción (las estaciones no se
  mueven); para el candidato dice si el patrón de uso se mantiene de un año a otro.
- **Sin estandarizar el perfil.** Todas las columnas son fracciones en la misma escala;
  estandarizar inflaría combinaciones con muy pocos viajes, como la madrugada.

### Modelos candidatos siguientes

`AgglomerativeClustering` (permite grupos dentro de grupos para planear por niveles) y
`GaussianMixture` (da la probabilidad de pertenecer a cada grupo, útil para estaciones que
mezclan usos).

## 7. Validación de datos con DeepChecks: qué bloquea y qué solo se vigila

- **Fecha:** 2026-11-01
- **Estado:** vigente

### Contexto

`scripts/validar_datos_p3.py` corre las suites `data_integrity` y `train_test_validation` de
DeepChecks sobre `data/processed/p3_perfiles.parquet`, con el entorno aparte
`.venv-deepchecks` (ver `requirements-deepchecks.txt`). Con las condiciones por defecto, la
primera corrida dio tres hallazgos (`reports/deepchecks/p3_hallazgos.txt`); pasaron los de
nulos, duplicados, tipos y columnas constantes.

### Decisión

| Hallazgo (condiciones por defecto) | Decisión |
|---|---|
| **Correlación entre variables:** 48 pares con correlación > 0.9 | Se acepta por diseño. La condición se cambia por una que solo falla con pares que el diseño no explica |
| **Drift por variable:** `sal_jue_noche` (KS 0.22) y `lle_sab_noche` (KS 0.23), umbral 0.2 | Se acepta como hallazgo menor y se reporta como advertencia; no se sube el umbral |
| **Drift multivariado:** 0.49 (AUC 0.74) entre entrenamiento y prueba, umbral 0.25 | Se documenta como limitación, se reporta como advertencia y se vigila con la estabilidad del modelo |

Los chequeos de integridad **bloquean**: si una condición falla, el script termina con código
1. Los de drift **solo advierten**: se muestran en la salida y en el reporte HTML, pero no
detienen el pipeline.

### Por qué

- **Las 48 correlaciones salen del diseño del perfil (decisión 2).** 47 son la misma franja en
  dos días entre semana (por ejemplo `sal_lun_manana` y `sal_mar_manana`): una estación que se
  usa para ir al trabajo se usa igual de lunes a viernes. La otra es `salidas_dia` con
  `llegadas_dia`: casi todas las bicicletas que salen de una estación regresan a ella. Con la
  condición por defecto el reporte fallaría siempre y dejaría de leerse; la nueva sigue
  fallando si aparece una correlación que no corresponde, por ejemplo por un error en
  `perfil_de_uso`. Costo aceptado: para KMeans, esas cinco columnas casi iguales por franja
  hacen que el patrón entre semana pese más en la distancia que el de fin de semana, lo cual
  es razonable porque son 5 de los 7 días.
- **El drift por variable es pequeño en la práctica.** La mediana de `sal_jue_noche` pasa de
  0.049 en entrenamiento a 0.044 en prueba: medio punto porcentual de los viajes. Subir el
  umbral solo para que pase ocultaría un cambio mayor más adelante.
- **El drift multivariado es real, no un artefacto.** Un clasificador distingue los perfiles
  de entrenamiento de los de prueba con AUC 0.74. No se explica por las estaciones nuevas: con
  solo las 365 que están en ambos periodos el drift es 0.56 (AUC 0.78). También aparece entre
  periodos vecinos (validación contra prueba: 0.35). Es decir, las mismas estaciones cambiaron
  su patrón de uso. Dos causas probables: entrenamiento promedia 3.5 años (2022–2025) y prueba
  solo cubre abril–septiembre de 2026 (estacionalidad), y el uso del sistema cambió en ese
  tiempo. Además, las 64 estaciones que solo tienen perfil en prueba (decisión 3) son muy
  distintas (mediana de 4.9 salidas al día contra 24.8 de las demás) y reciben un grupo
  aprendido sin ellas.
- **El drift no detiene el pipeline porque ya hay métricas que miden su efecto.**
  `test_ari_estabilidad` y `test_silhouette` (decisión 6) dicen si los grupos siguen siendo
  útiles en prueba. Si un pipeline falla siempre por drift esperado, el equipo aprende a
  ignorarlo.

### Pendiente

- Probar entrenar con una ventana reciente (por ejemplo, los últimos 12 meses) para que el
  perfil de entrenamiento se parezca más al de prueba. Cambia la decisión 4 y obliga a
  reentrenar y registrar de nuevo.
- Definir un umbral de drift multivariado a partir del cual se reentrena (monitoreo).
