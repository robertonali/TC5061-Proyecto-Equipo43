"""Modelos de P2 registrados en MLflow.

Lee la tabla de `features.py` (data/processed/p2_ventanas.parquet), la parte por fechas
(decisión 6), entrena el modelo elegido con `--modelo` y lo evalúa en validación y prueba
(decisiones 3, 7 y 8):

- `mayoritaria` y `frecuencia`: los baselines (clase mayoritaria y frecuencia de cada clase en
  las últimas 4 semanas, misma estación, día de la semana y ventana);
- `logistica`: regresión logística, con las variables preparadas en un Pipeline;
- `hgb`: HistGradientBoostingClassifier;
- `busqueda-hgb`: búsqueda de hiperparámetros de `hgb` en la rejilla P2_BUSQUEDA_HGB, como un run
  padre con un run hijo (nested run) por combinación. Los hijos solo se evalúan en validación.

Los hiperparámetros por defecto están en P2_HIPERPARAMETROS (config.py) y se cambian con las
opciones de la terminal (o con `-P` en `mlflow run`).

Cada run registra las fechas, las versiones de los datos, la semilla y los hiperparámetros como
parámetros; recall, precisión y PR-AUC de cada clase de riesgo como métricas `val_*` y `test_*`,
más su promedio (`*_pr_auc_promedio`, la métrica para elegir modelo), su variación de un mes a
otro y el tiempo de entrenamiento y predicción; el equipo, la pregunta, el commit, la familia del
modelo (`model_family`) y la etapa del trabajo (`stage_of_work`) como etiquetas. Como artefactos:
gráficas en `plots/`, el manifiesto de los datos en `data/`, la configuración en `config/` y, en
los candidatos, el modelo entrenado con su firma y un ejemplo de entrada.

Uso (normalmente con `mlflow run`, ver la sección de P2 del README):
    python -m mibici_mlops.modeling.train_p2 --modelo frecuencia
    python -m mibici_mlops.modeling.train_p2 --modelo hgb --learning-rate 0.05
    python -m mibici_mlops.modeling.train_p2 --modelo busqueda-hgb

Antes de correrlo, el servidor de MLflow debe estar levantado (ver la sección de MLflow del
README):
    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination mlruns --port 5000 --workers 1

Los runs se ven en http://127.0.0.1:5000.
"""

import itertools
import json
from pathlib import Path
import time

from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mibici_mlops.config import (
    EXPERIMENTO_MLFLOW,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    MLFLOW_TRACKING_URI,
    P2_BUSQUEDA_HGB,
    P2_HIPERPARAMETROS,
    P2_HORA_REINICIO,
    P2_INVENTARIO_INICIAL,
    P2_SEMANAS_HISTORIA,
    P2_VENTANA,
    P2_VENTANAS_PATH,
    PROJ_ROOT,
    SEMILLA,
)
from mibici_mlops.features import CLASES_P2
from mibici_mlops.modeling.registro import (
    etiquetas_run,
    registrar_manifiesto,
    usar_experimento,
    version_datos,
)

app = typer.Typer()

CLASES_RIESGO = ["se_vacia", "se_llena"]
NOMBRES = {"se_vacia": "Se vacía", "se_llena": "Se llena", "sin_riesgo": "Sin riesgo"}
# Conjuntos en que se evalúa cada modelo: prefijo de sus métricas → nombre en las gráficas.
NOMBRES_CONJUNTO = {"val": "validación", "test": "prueba"}

# Variables de los candidatos (decisión 8): todas se conocen a las 4:00, cuando se predice.
# frec_sin_riesgo no entra porque es 1 menos las otras dos frecuencias.
VARIABLES = ["hora", "dia_semana", "feriado", "capacidad", "frec_se_vacia", "frec_se_llena"]
CATEGORICAS = ["hora", "dia_semana"]

# MLflow guarda los modelos de scikit-learn con skops, que al cargarlos rechaza los tipos que no
# conoce. Estos son parte de HistGradientBoostingClassifier y se declaran aquí, uno por uno, en
# lugar de volver a pickle, que puede ejecutar código arbitrario al cargar el modelo.
TIPOS_CONFIABLES_SKOPS = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]

# Archivos de configuración que se guardan en config/ de cada run.
ARCHIVOS_CONFIGURACION = [
    "MLproject",
    "python_env.yaml",
    "requirements.txt",
    "mibici_mlops/config.py",
]

# Colores de las gráficas: espacios 1 y 2 de la paleta categórica de referencia (validada para
# daltonismo) y su rampa secuencial azul para la matriz de confusión; el texto usa tinta neutra,
# nunca el color de la serie.
COLORES = {"se_vacia": "#2a78d6", "se_llena": "#eb6834"}
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONDO = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
CUADRICULA = "#e6e5e1"


def asignar_conjunto(dia: pd.Series) -> pd.Series:
    """Conjunto de cada ventana según su día: entrenamiento, validación o prueba (decisión 6)."""
    conjunto = pd.Series("entrenamiento", index=dia.index)
    conjunto[dia >= pd.Timestamp(FECHA_INICIO_VALIDACION)] = "validacion"
    conjunto[dia >= pd.Timestamp(FECHA_INICIO_PRUEBA)] = "prueba"
    return conjunto


def baseline_clase_mayoritaria(
    entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame
) -> pd.DataFrame:
    """Probabilidad de cada clase con `DummyClassifier(strategy="most_frequent")`.

    Aprende la clase más común en entrenamiento ("sin riesgo") y la predice siempre. Es el piso:
    no detecta ningún riesgo y aun así tiene exactitud alta, por eso la exactitud no sirve como
    métrica. No usa variables: recibe una columna de ceros porque scikit-learn pide una matriz.
    """
    modelo = DummyClassifier(strategy="most_frequent", random_state=SEMILLA)
    modelo.fit(np.zeros((len(entrenamiento), 1)), entrenamiento["clase"])
    probabilidades = modelo.predict_proba(np.zeros((len(evaluacion), 1)))
    return pd.DataFrame(probabilidades, columns=modelo.classes_, index=evaluacion.index)[CLASES_P2]


def baseline_frecuencia(entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada clase: su frecuencia en las últimas P2_SEMANAS_HISTORIA semanas.

    No se entrena: usa las columnas `frec_*` de features.py, que solo miran semanas anteriores.
    Recibe `entrenamiento` aunque no lo use, para que los dos baselines se llamen igual.
    """
    columnas = [f"frec_{clase}" for clase in CLASES_P2]
    return evaluacion[columnas].set_axis(CLASES_P2, axis=1)


def pesos_de_clase(class_weight: str) -> str | None:
    """Traduce la opción `class_weight` ("balanced" o "none") al valor que pide scikit-learn."""
    if class_weight not in ("balanced", "none"):
        raise typer.BadParameter("class_weight debe ser balanced o none")
    return None if class_weight == "none" else class_weight


def crear_logistica(c: float, max_iter: int, class_weight: str) -> Pipeline:
    """Regresión logística con las variables preparadas en un Pipeline (decisión 8).

    `hora` y `dia_semana` pasan por OneHotEncoder porque son categorías, no cantidades, y
    `capacidad` por StandardScaler para quedar en una escala parecida a la de las frecuencias (0 a
    1). La preparación aprende solo del entrenamiento y viaja con el modelo: quien lo cargue del
    Model Registry le pasa las columnas tal como salen de features.py.

    Por qué para P2: es el candidato más simple e interpretable; si un modelo más complejo no lo
    supera con claridad, no vale su costo.
    """
    preparacion = ColumnTransformer(
        [
            ("categorias", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("escala", StandardScaler(), ["capacidad"]),
            ("sin_cambio", "passthrough", ["feriado", "frec_se_vacia", "frec_se_llena"]),
        ]
    )
    clasificador = LogisticRegression(
        C=c, max_iter=max_iter, class_weight=pesos_de_clase(class_weight)
    )
    return Pipeline([("preparacion", preparacion), ("clasificador", clasificador)])


def crear_hgb(
    learning_rate: float, max_iter: int, max_leaf_nodes: int, class_weight: str
) -> Pipeline:
    """HistGradientBoostingClassifier en un Pipeline (decisión 8).

    No hace falta preparar las variables: los árboles cortan por umbrales, así que la escala no
    importa, y `hora` y `dia_semana` se declaran categóricas para que un solo corte pueda separar,
    por ejemplo, las horas pico del resto. Sin early stopping, porque apartaría al azar una parte
    del entrenamiento para decidir cuándo parar y mezclaría fechas; el número de árboles
    (`max_iter`) es un hiperparámetro fijo.

    Por qué para P2: captura combinaciones (hora pico de un día hábil en una estación que suele
    vaciarse) que la regresión logística no ve, y entrena rápido con millones de ventanas.
    """
    clasificador = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        class_weight=pesos_de_clase(class_weight),
        categorical_features=CATEGORICAS,
        early_stopping=False,
        random_state=SEMILLA,
    )
    return Pipeline([("clasificador", clasificador)])


# Modelos que se pueden correr con --modelo; cada llamada registra un run. Los baselines dan
# directamente la probabilidad de cada clase; los candidatos se crean con sus hiperparámetros
# (P2_HIPERPARAMETROS y la terminal) y se entrenan una sola vez.
MODELOS = {
    "mayoritaria": {
        "familia": "dummy",
        "etapa": "baseline",
        "probabilidades": baseline_clase_mayoritaria,
        "hiperparametros": {"modelo": "DummyClassifier", "strategy": "most_frequent"},
    },
    "frecuencia": {
        "familia": "heuristica",
        "etapa": "baseline",
        "probabilidades": baseline_frecuencia,
        "hiperparametros": {
            "modelo": "frecuencia_semanas_anteriores",
            "semanas": P2_SEMANAS_HISTORIA,
            "desempate": " > ".join(CLASES_P2),
        },
    },
    "logistica": {"familia": "lineal", "etapa": "candidate", "crear": crear_logistica},
    "hgb": {"familia": "gradient_boosting", "etapa": "candidate", "crear": crear_hgb},
}
BUSQUEDA = "busqueda-hgb"


def resolver_hiperparametros(modelo: str, opciones: dict) -> dict:
    """Hiperparámetros del modelo: los de P2_HIPERPARAMETROS, con los de la terminal encima.

    Falla si se pasa uno que el modelo no usa, para que un error de dedo no se ignore en silencio.
    En la búsqueda, la rejilla define los suyos y no se pueden pasar por la terminal.
    """
    if modelo == BUSQUEDA:
        por_defecto = {
            nombre: valor
            for nombre, valor in P2_HIPERPARAMETROS["hgb"].items()
            if nombre not in P2_BUSQUEDA_HGB
        }
    else:
        por_defecto = P2_HIPERPARAMETROS.get(modelo, {})
    pasados = {nombre: valor for nombre, valor in opciones.items() if valor is not None}
    sobrantes = set(pasados) - set(por_defecto)
    if sobrantes:
        raise typer.BadParameter(f"{modelo} no acepta: {', '.join(sorted(sobrantes))}")
    return {**por_defecto, **pasados}


def nombre_run(modelo: str, hiperparametros: dict) -> str:
    """Nombre descriptivo del run: el modelo y los hiperparámetros que lo distinguen."""
    if modelo == "mayoritaria":
        return "p2-baseline-clase-mayoritaria"
    if modelo == "frecuencia":
        return f"p2-baseline-frecuencia-{P2_SEMANAS_HISTORIA}-semanas"
    pesos = "balanceado" if hiperparametros["class_weight"] == "balanced" else "sin-pesos"
    if modelo == "logistica":
        return f"p2-logistica-{pesos}-c{hiperparametros['c']}"
    if modelo == BUSQUEDA:
        return f"p2-busqueda-hgb-{pesos}-arboles{hiperparametros['max_iter']}"
    return (
        f"p2-hgb-{pesos}-lr{hiperparametros['learning_rate']}"
        f"-hojas{hiperparametros['max_leaf_nodes']}-arboles{hiperparametros['max_iter']}"
    )


def entrenar(modelo: Pipeline, entrenamiento: pd.DataFrame) -> tuple[Pipeline, float]:
    """Entrena el modelo con el conjunto de entrenamiento y regresa los segundos que tardó."""
    inicio = time.perf_counter()
    modelo.fit(entrenamiento[VARIABLES], entrenamiento["clase"])
    return modelo, time.perf_counter() - inicio


def probabilidades_modelo(modelo: Pipeline, datos: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada clase según un modelo entrenado, en el orden de CLASES_P2."""
    probabilidades = modelo.predict_proba(datos[VARIABLES])
    return pd.DataFrame(probabilidades, columns=modelo.classes_, index=datos.index)[CLASES_P2]


def predecir(probabilidades: pd.DataFrame) -> pd.Series:
    """Clase con mayor probabilidad. En empate gana la primera de CLASES_P2 (decisión 7)."""
    indice = probabilidades[CLASES_P2].to_numpy().argmax(axis=1)
    return pd.Series(np.array(CLASES_P2)[indice], index=probabilidades.index)


def calcular_metricas(
    clase: pd.Series, probabilidades: pd.DataFrame, prefijo: str
) -> dict[str, float]:
    """Recall, precisión y PR-AUC de cada clase de riesgo, con el prefijo del conjunto (val, test).

    También registra el promedio de las dos PR-AUC, la métrica para elegir modelo (decisión 8), y
    la proporción de cada clase: es la PR-AUC que tendría un modelo al azar, y sirve para leer las
    demás métricas.
    """
    prediccion = predecir(probabilidades)
    metricas = {}
    for riesgo in CLASES_RIESGO:
        real = clase == riesgo
        predicha = prediccion == riesgo
        metricas[f"{prefijo}_recall_{riesgo}"] = recall_score(real, predicha)
        metricas[f"{prefijo}_precision_{riesgo}"] = precision_score(
            real, predicha, zero_division=0
        )
        metricas[f"{prefijo}_pr_auc_{riesgo}"] = average_precision_score(
            real, probabilidades[riesgo]
        )
        metricas[f"{prefijo}_proporcion_{riesgo}"] = real.mean()
    metricas[f"{prefijo}_pr_auc_promedio"] = float(
        np.mean([metricas[f"{prefijo}_pr_auc_{riesgo}"] for riesgo in CLASES_RIESGO])
    )
    return metricas


def pr_auc_por_mes(
    conjuntos: dict[str, pd.DataFrame], probabilidades: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """PR-AUC de cada clase de riesgo y su promedio en cada mes de validación y de prueba.

    Por qué para P2: un modelo que gana en el total pero sube y baja mucho de un mes a otro es
    menos confiable para el equipo de rebalanceo. Su desviación estándar mide la estabilidad, uno
    de los criterios para elegir el modelo.
    """
    filas = []
    for prefijo, datos in conjuntos.items():
        for mes, grupo in datos.groupby(datos["dia"].dt.to_period("M")):
            fila = {"conjunto": prefijo, "mes": mes.to_timestamp()}
            for riesgo in CLASES_RIESGO:
                fila[riesgo] = average_precision_score(
                    grupo["clase"] == riesgo, probabilidades[prefijo].loc[grupo.index, riesgo]
                )
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla["promedio"] = tabla[CLASES_RIESGO].mean(axis=1)
    return tabla


def dar_estilo(eje) -> None:
    """Fondo, cuadrícula y ejes discretos, iguales en todas las gráficas."""
    eje.set_facecolor(FONDO)
    eje.grid(color=CUADRICULA, linewidth=0.8)
    eje.set_axisbelow(True)
    eje.tick_params(colors=TEXTO_SECUNDARIO)
    eje.spines[["top", "right"]].set_visible(False)
    eje.spines[["left", "bottom"]].set_color(CUADRICULA)


def graficar_curvas_pr(clase: pd.Series, probabilidades: pd.DataFrame, titulo: str) -> Figure:
    """Curva precisión-recall de cada clase de riesgo, con su proporción como referencia del azar.

    El punto marca la decisión del modelo (la clase más probable); no aparece si el modelo nunca
    predice esa clase.
    """
    prediccion = predecir(probabilidades)
    figura = Figure(figsize=(10, 4.2), facecolor=FONDO)
    ejes = figura.subplots(1, 2, sharey=True)
    for eje, riesgo in zip(ejes, CLASES_RIESGO):
        real = clase == riesgo
        precision, recall, _ = precision_recall_curve(real, probabilidades[riesgo])
        pr_auc = average_precision_score(real, probabilidades[riesgo])
        eje.step(recall, precision, where="post", color=COLORES[riesgo], linewidth=1.5)

        eje.axhline(real.mean(), color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
        eje.text(
            0.01,
            real.mean(),
            f" azar: {real.mean():.1%}",
            transform=eje.get_yaxis_transform(),
            va="bottom",
            color=TEXTO_SECUNDARIO,
            fontsize=9,
        )
        predicha = prediccion == riesgo
        if predicha.any():
            punto = (recall_score(real, predicha), precision_score(real, predicha))
            eje.plot(*punto, "o", color=COLORES[riesgo], markersize=8, markeredgecolor=FONDO)
            eje.annotate(
                "decisión del modelo",
                punto,
                xytext=(8, 6),
                textcoords="offset points",
                color=TEXTO_SECUNDARIO,
                fontsize=9,
            )

        eje.set_title(f"{NOMBRES[riesgo]} · PR-AUC {pr_auc:.3f}", loc="left", color=TEXTO)
        eje.set_xlabel("Recall", color=TEXTO_SECUNDARIO)
        eje.set_xlim(0, 1)
        eje.set_ylim(0, 1.02)
        dar_estilo(eje)
    ejes[0].set_ylabel("Precisión", color=TEXTO_SECUNDARIO)
    figura.suptitle(titulo, x=0.01, ha="left", color=TEXTO)
    figura.tight_layout()
    return figura


def graficar_matriz_confusion(
    clase: pd.Series, probabilidades: pd.DataFrame, titulo: str
) -> Figure:
    """Matriz de confusión normalizada por fila: qué parte de cada clase real se predijo como cuál.

    La diagonal es el recall de cada clase. Cada celda lleva también el número de ventanas.
    """
    conteos = confusion_matrix(clase, predecir(probabilidades), labels=CLASES_P2)
    fracciones = conteos / conteos.sum(axis=1, keepdims=True)
    figura = Figure(figsize=(6.4, 5.2), facecolor=FONDO)
    eje = figura.subplots()
    rampa = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    eje.imshow(fracciones, cmap=rampa, vmin=0, vmax=1)
    for fila, columna in itertools.product(range(len(CLASES_P2)), repeat=2):
        valor = fracciones[fila, columna]
        eje.text(
            columna,
            fila,
            f"{valor:.0%}\n({conteos[fila, columna]:,})",
            ha="center",
            va="center",
            color="#ffffff" if valor > 0.5 else TEXTO,
            fontsize=10,
        )
    etiquetas = [NOMBRES[c] for c in CLASES_P2]
    eje.set_xticks(range(len(CLASES_P2)), etiquetas, color=TEXTO_SECUNDARIO)
    eje.set_yticks(range(len(CLASES_P2)), etiquetas, color=TEXTO_SECUNDARIO)
    eje.set_xlabel("Clase predicha", color=TEXTO_SECUNDARIO)
    eje.set_ylabel("Clase real", color=TEXTO_SECUNDARIO)
    eje.tick_params(length=0)
    eje.spines[:].set_visible(False)
    eje.set_title(titulo, loc="left", color=TEXTO, fontsize=11)
    figura.tight_layout()
    return figura


def graficar_estabilidad(por_mes: pd.DataFrame, titulo: str) -> Figure:
    """PR-AUC de cada clase de riesgo por mes, en validación y prueba (ver `pr_auc_por_mes`)."""
    figura = Figure(figsize=(10, 4.2), facecolor=FONDO)
    eje = figura.subplots()
    ultimo = por_mes.iloc[-1]
    # Etiqueta al final de cada línea; si los dos valores quedan cerca, se separan para no
    # encimarse.
    alturas = {riesgo: ultimo[riesgo] for riesgo in CLASES_RIESGO}
    if abs(alturas["se_vacia"] - alturas["se_llena"]) < 0.06:
        centro = (alturas["se_vacia"] + alturas["se_llena"]) / 2
        arriba = max(CLASES_RIESGO, key=alturas.get)
        alturas = {r: centro + (0.03 if r == arriba else -0.03) for r in CLASES_RIESGO}
    for riesgo in CLASES_RIESGO:
        eje.plot(
            por_mes["mes"],
            por_mes[riesgo],
            color=COLORES[riesgo],
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=FONDO,
            label=NOMBRES[riesgo],
        )
        eje.annotate(
            NOMBRES[riesgo],
            (ultimo["mes"], alturas[riesgo]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=TEXTO_SECUNDARIO,
            fontsize=9,
        )
    eje.set_xlim(right=ultimo["mes"] + pd.Timedelta(days=40))
    corte = pd.Timestamp(FECHA_INICIO_PRUEBA) - pd.Timedelta(days=15)
    eje.axvline(corte, color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
    for texto, alineacion in [("validación ", "right"), (" prueba", "left")]:
        eje.text(
            corte,
            0.97,
            texto,
            transform=eje.get_xaxis_transform(),
            ha=alineacion,
            va="top",
            color=TEXTO_SECUNDARIO,
            fontsize=9,
        )
    eje.set_ylim(0, 1)
    eje.set_ylabel("PR-AUC", color=TEXTO_SECUNDARIO)
    eje.legend(loc="lower left", frameon=False, labelcolor=TEXTO_SECUNDARIO)
    dar_estilo(eje)
    eje.set_title(titulo, loc="left", color=TEXTO)
    figura.tight_layout()
    return figura


def registrar_parametros(parametros: dict) -> None:
    """Registra los parámetros del run activo sin repetir los que ya registró `mlflow run`.

    `mlflow run` guarda como texto los -P que recibe (p. ej. "0.10"); volver a registrarlos con
    otro formato (0.1) haría fallar a MLflow, que no deja cambiar el valor de un parámetro.
    """
    ya_registrados = mlflow.get_run(mlflow.active_run().info.run_id).data.params
    mlflow.log_params({k: v for k, v in parametros.items() if k not in ya_registrados})


def registrar_evaluacion(
    nombre: str, conjuntos: dict[str, pd.DataFrame], probabilidades: dict[str, pd.DataFrame]
) -> dict[str, float]:
    """Métricas de validación y prueba, su estabilidad por mes y las gráficas en plots/."""
    metricas = {}
    for prefijo, datos in conjuntos.items():
        metricas.update(calcular_metricas(datos["clase"], probabilidades[prefijo], prefijo))
        conjunto = NOMBRES_CONJUNTO[prefijo]
        mlflow.log_figure(
            graficar_curvas_pr(
                datos["clase"], probabilidades[prefijo], f"P2 · {nombre} (conjunto de {conjunto})"
            ),
            f"plots/curva_precision_recall_{prefijo}.png",
        )
        mlflow.log_figure(
            graficar_matriz_confusion(
                datos["clase"], probabilidades[prefijo], f"P2 · {nombre}\nConjunto de {conjunto}"
            ),
            f"plots/matriz_confusion_{prefijo}.png",
        )
    por_mes = pr_auc_por_mes(conjuntos, probabilidades)
    for prefijo in conjuntos:
        promedios = por_mes.loc[por_mes["conjunto"] == prefijo, "promedio"]
        metricas[f"{prefijo}_pr_auc_promedio_desv_mensual"] = promedios.std()
    mlflow.log_figure(
        graficar_estabilidad(por_mes, f"P2 · {nombre} · PR-AUC por mes"),
        "plots/pr_auc_por_mes.png",
    )
    mlflow.log_metrics(metricas)
    return metricas


def ruta_para_registro(ruta: Path) -> str:
    """Ruta relativa al proyecto (o solo el nombre del archivo si está fuera de él).

    Así no se registra en MLflow la carpeta personal de quien corre el run.
    """
    try:
        return ruta.resolve().relative_to(PROJ_ROOT).as_posix()
    except ValueError:
        return ruta.name


def registrar_datos(
    entrenamiento: pd.DataFrame, conjuntos: dict[str, pd.DataFrame], fuente: str
) -> None:
    """Versión de los datos en data/: el manifiesto de la ingesta y los conjuntos con log_input.

    `log_input` guarda el esquema, el número de filas y una huella (digest) de cada conjunto, no
    los datos. La tabla de P2 no tiene datos de los usuarios (decisión 7); `fuente` es la tabla de
    la que salen, como la da `ruta_para_registro`.
    """
    registrar_manifiesto()
    contextos = {
        "training": entrenamiento,
        "validation": conjuntos["val"],
        "testing": conjuntos["test"],
    }
    for contexto, datos in contextos.items():
        dataset = mlflow.data.from_pandas(
            datos[VARIABLES + ["clase"]],
            source=fuente,
            name=f"p2_ventanas_{contexto}",
            targets="clase",
        )
        mlflow.log_input(dataset, context=contexto)


def registrar_configuracion(configuracion: dict) -> None:
    """Configuración con que corrió el run en config/: la del run y los archivos del proyecto."""
    mlflow.log_dict(configuracion, "config/run_p2.json")
    for archivo in ARCHIVOS_CONFIGURACION:
        mlflow.log_artifact(str(PROJ_ROOT / archivo), artifact_path="config")


def registrar_modelo(modelo: Pipeline, entrenamiento: pd.DataFrame) -> None:
    """Guarda el modelo entrenado con su firma (columnas y tipos de entrada y salida) y un ejemplo.

    La firma deja claro qué columnas espera el modelo; quien lo cargue del Model Registry puede
    usar `predict` (la clase) o, con `mlflow.sklearn.load_model`, `predict_proba`.
    """
    ejemplo = entrenamiento[VARIABLES].head(5)
    firma = infer_signature(ejemplo, modelo.predict(ejemplo))
    mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        signature=firma,
        input_example=ejemplo,
        skops_trusted_types=TIPOS_CONFIABLES_SKOPS,
    )


def correr_modelo(
    modelo: str,
    hiperparametros: dict,
    entrenamiento: pd.DataFrame,
    conjuntos: dict[str, pd.DataFrame],
    parametros: dict,
    etiquetas: dict,
) -> dict[str, float]:
    """Entrena (si hace falta), evalúa y registra un modelo como un run de MLflow."""
    elegido = MODELOS[modelo]
    nombre = nombre_run(modelo, hiperparametros)
    tags = {**etiquetas, "model_family": elegido["familia"], "stage_of_work": elegido["etapa"]}
    with mlflow.start_run(run_name=nombre, tags=tags):
        mlflow.set_tag("mlflow.runName", nombre)
        if "crear" in elegido:
            estimador, segundos = entrenar(elegido["crear"](**hiperparametros), entrenamiento)
            inicio = time.perf_counter()
            probabilidades = {
                prefijo: probabilidades_modelo(estimador, datos)
                for prefijo, datos in conjuntos.items()
            }
            tiempos = {
                "tiempo_entrenamiento_s": segundos,
                "tiempo_prediccion_s": time.perf_counter() - inicio,
            }
            hiperparametros = {
                "modelo": type(estimador[-1]).__name__,
                "variables": ",".join(VARIABLES),
                **hiperparametros,
            }
        else:
            estimador, tiempos = None, {}
            probabilidades = {
                prefijo: elegido["probabilidades"](entrenamiento, datos)
                for prefijo, datos in conjuntos.items()
            }
            hiperparametros = elegido["hiperparametros"]
        registrar_parametros({**parametros, **hiperparametros})
        metricas = registrar_evaluacion(nombre, conjuntos, probabilidades)
        mlflow.log_metrics(tiempos)
        registrar_datos(entrenamiento, conjuntos, parametros["tabla"])
        registrar_configuracion(
            {"modelo": modelo, "hiperparametros": hiperparametros, "parametros": parametros}
        )
        if estimador is not None:
            registrar_modelo(estimador, entrenamiento)
    resumen = {k: round(v, 3) for k, v in {**metricas, **tiempos}.items() if "proporcion" not in k}
    logger.info(f"{nombre}: {resumen}")
    return metricas


def buscar_hgb(
    fijos: dict,
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    parametros: dict,
    etiquetas: dict,
) -> pd.DataFrame:
    """Búsqueda de hiperparámetros de `hgb` en la rejilla P2_BUSQUEDA_HGB, con nested runs.

    El run padre agrupa la búsqueda y guarda la mejor combinación y la tabla de resultados; cada
    combinación es un run hijo. Los hijos solo se evalúan en validación: la prueba se reserva para
    el candidato final, que se vuelve a correr con `--modelo hgb` y los mejores hiperparámetros.
    """
    combinaciones = [
        dict(zip(P2_BUSQUEDA_HGB, valores))
        for valores in itertools.product(*P2_BUSQUEDA_HGB.values())
    ]
    nombre = nombre_run(BUSQUEDA, fijos)
    tags = {**etiquetas, "model_family": "gradient_boosting", "stage_of_work": "search"}
    resultados = []
    with mlflow.start_run(run_name=nombre, tags=tags):
        mlflow.set_tag("mlflow.runName", nombre)
        registrar_parametros(
            {
                **parametros,
                **fijos,
                "modelo": "HistGradientBoostingClassifier",
                "variables": ",".join(VARIABLES),
                "rejilla": json.dumps(P2_BUSQUEDA_HGB),
                "combinaciones": len(combinaciones),
            }
        )
        for numero, combinacion in enumerate(combinaciones, start=1):
            hiperparametros = {**fijos, **combinacion}
            hijo = nombre_run("hgb", hiperparametros)
            logger.info(f"Combinación {numero}/{len(combinaciones)}: {hijo}")
            with mlflow.start_run(run_name=hijo, nested=True, tags=tags):
                mlflow.log_params(
                    {**parametros, **hiperparametros, "modelo": "HistGradientBoostingClassifier"}
                )
                estimador, segundos = entrenar(crear_hgb(**hiperparametros), entrenamiento)
                metricas = calcular_metricas(
                    validacion["clase"], probabilidades_modelo(estimador, validacion), "val"
                )
                mlflow.log_metrics({**metricas, "tiempo_entrenamiento_s": segundos})
            resultados.append({**combinacion, **metricas, "tiempo_entrenamiento_s": segundos})

        tabla = pd.DataFrame(resultados).sort_values("val_pr_auc_promedio", ascending=False)
        # Columna por columna para que cada valor conserve su tipo (63, no 63.0).
        mejor = {clave: tabla[clave].iloc[0].item() for clave in P2_BUSQUEDA_HGB}
        mlflow.log_metric("mejor_val_pr_auc_promedio", tabla["val_pr_auc_promedio"].iloc[0])
        mlflow.log_params({f"mejor_{clave}": valor for clave, valor in mejor.items()})
        mlflow.log_text(tabla.to_csv(index=False), "busqueda/resultados.csv")
        registrar_manifiesto()
        registrar_configuracion(
            {
                "modelo": BUSQUEDA,
                "fijos": fijos,
                "rejilla": P2_BUSQUEDA_HGB,
                "parametros": parametros,
            }
        )
    logger.info(f"Mejor combinación: {mejor}")
    return tabla


@app.command()
def main(
    modelo: str = typer.Option(..., help=f"Modelo a correr: {', '.join([*MODELOS, BUSQUEDA])}."),
    c: float = typer.Option(None, help="logistica: inverso de la regularización."),
    max_iter: int = typer.Option(None, help="logistica: iteraciones; hgb: número de árboles."),
    class_weight: str = typer.Option(None, help="logistica y hgb: balanced o none."),
    learning_rate: float = typer.Option(None, help="hgb: tasa de aprendizaje."),
    max_leaf_nodes: int = typer.Option(None, help="hgb: hojas máximas por árbol."),
    ventanas_path: Path = P2_VENTANAS_PATH,
):
    if modelo not in [*MODELOS, BUSQUEDA]:
        raise typer.BadParameter(
            f"usa uno de: {', '.join([*MODELOS, BUSQUEDA])}", param_hint="--modelo"
        )
    opciones = {
        "c": c,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "learning_rate": learning_rate,
        "max_leaf_nodes": max_leaf_nodes,
    }
    hiperparametros = resolver_hiperparametros(modelo, opciones)

    tabla = pd.read_parquet(ventanas_path)
    tabla["conjunto"] = asignar_conjunto(tabla["dia"])
    entrenamiento = tabla[tabla["conjunto"] == "entrenamiento"]
    conjuntos = {
        "val": tabla[tabla["conjunto"] == "validacion"],
        "test": tabla[tabla["conjunto"] == "prueba"],
    }
    logger.info(
        f"Ventanas: {len(entrenamiento):,} de entrenamiento, {len(conjuntos['val']):,} de "
        f"validación y {len(conjuntos['test']):,} de prueba"
    )

    usar_experimento()
    etiquetas = etiquetas_run("P2", __file__)
    parametros = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_inicio_validacion": FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": FECHA_INICIO_PRUEBA,
        "version_datos": version_datos(),
        "tabla": ruta_para_registro(ventanas_path),
        "semilla": SEMILLA,
        "ventana": P2_VENTANA,
        "hora_reinicio": P2_HORA_REINICIO,
        "inventario_inicial": P2_INVENTARIO_INICIAL,
        "ventanas_entrenamiento": len(entrenamiento),
        "ventanas_validacion": len(conjuntos["val"]),
        "ventanas_prueba": len(conjuntos["test"]),
    }

    if modelo == BUSQUEDA:
        buscar_hgb(hiperparametros, entrenamiento, conjuntos["val"], parametros, etiquetas)
    else:
        correr_modelo(modelo, hiperparametros, entrenamiento, conjuntos, parametros, etiquetas)
    logger.success(f"Registrado en el experimento {EXPERIMENTO_MLFLOW} ({MLFLOW_TRACKING_URI})")


if __name__ == "__main__":
    app()
