"""Baselines de P2 registrados en MLflow.

Lee la tabla de `features.py` (data/processed/p2_ventanas.parquet), la parte por fechas
(decisión 6) y evalúa en el conjunto de prueba dos baselines (decisiones 3 y 7):

- clase mayoritaria (`DummyClassifier`), que siempre predice "sin riesgo";
- frecuencia de cada clase en las últimas 4 semanas (misma estación, día de la semana y ventana).

Cada baseline es un run del experimento EXPERIMENTO_MLFLOW con: las fechas, la versión de los
datos, la semilla y los hiperparámetros como parámetros; recall, precisión y PR-AUC de cada clase
de riesgo como métricas `test_*`; el equipo, la pregunta y el commit como etiquetas, y la curva
precisión-recall como artefacto.

Uso:
    python -m mibici_mlops.modeling.train_p2

Antes de correrlo, el servidor de MLflow debe estar levantado (ver la sección de MLflow del
README):
    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination mlruns --port 5000 --workers 1

Los runs se ven en http://127.0.0.1:5000.
"""

from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
import typer

from mibici_mlops.config import (
    EXPERIMENTO_MLFLOW,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    MLFLOW_TRACKING_URI,
    P2_HORA_REINICIO,
    P2_INVENTARIO_INICIAL,
    P2_SEMANAS_HISTORIA,
    P2_VENTANA,
    P2_VENTANAS_PATH,
    SEMILLA,
)
from mibici_mlops.features import CLASES_P2
from mibici_mlops.modeling.registro import etiquetas_run, usar_experimento, version_datos

app = typer.Typer()

CLASES_RIESGO = ["se_vacia", "se_llena"]
NOMBRES = {"se_vacia": "Se vacía", "se_llena": "Se llena"}

# Colores de la gráfica: espacios 1 y 2 de la paleta categórica de referencia (validada para
# daltonismo); el texto usa tinta neutra, nunca el color de la serie.
COLORES = {"se_vacia": "#2a78d6", "se_llena": "#eb6834"}
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


def baseline_clase_mayoritaria(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada clase con `DummyClassifier(strategy="most_frequent")`.

    Aprende la clase más común en entrenamiento ("sin riesgo") y la predice siempre. Es el piso:
    no detecta ningún riesgo y aun así tiene exactitud alta, por eso la exactitud no sirve como
    métrica. No usa variables: recibe una columna de ceros porque scikit-learn pide una matriz.
    """
    modelo = DummyClassifier(strategy="most_frequent", random_state=SEMILLA)
    modelo.fit(np.zeros((len(entrenamiento), 1)), entrenamiento["clase"])
    probabilidades = modelo.predict_proba(np.zeros((len(prueba), 1)))
    return pd.DataFrame(probabilidades, columns=modelo.classes_, index=prueba.index)[CLASES_P2]


def baseline_frecuencia(prueba: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada clase: su frecuencia en las últimas P2_SEMANAS_HISTORIA semanas.

    No se entrena: usa las columnas `frec_*` de features.py, que solo miran semanas anteriores.
    """
    columnas = [f"frec_{clase}" for clase in CLASES_P2]
    return prueba[columnas].set_axis(CLASES_P2, axis=1)


def predecir(probabilidades: pd.DataFrame) -> pd.Series:
    """Clase con mayor probabilidad. En empate gana la primera de CLASES_P2 (decisión 7)."""
    indice = probabilidades[CLASES_P2].to_numpy().argmax(axis=1)
    return pd.Series(np.array(CLASES_P2)[indice], index=probabilidades.index)


def calcular_metricas(clase: pd.Series, probabilidades: pd.DataFrame) -> dict[str, float]:
    """Recall, precisión y PR-AUC de cada clase de riesgo, con el prefijo del conjunto (test_).

    También registra la proporción de cada clase: es la PR-AUC que tendría un modelo al azar, y
    sirve para leer las demás métricas.
    """
    prediccion = predecir(probabilidades)
    metricas = {}
    for riesgo in CLASES_RIESGO:
        real = clase == riesgo
        predicha = prediccion == riesgo
        metricas[f"test_recall_{riesgo}"] = recall_score(real, predicha)
        metricas[f"test_precision_{riesgo}"] = precision_score(real, predicha, zero_division=0)
        metricas[f"test_pr_auc_{riesgo}"] = average_precision_score(real, probabilidades[riesgo])
        metricas[f"test_proporcion_{riesgo}"] = real.mean()
    return metricas


def graficar_curvas_pr(clase: pd.Series, probabilidades: pd.DataFrame, titulo: str) -> Figure:
    """Curva precisión-recall de cada clase de riesgo, con su proporción como referencia del azar.

    El punto marca la decisión del baseline (la clase más probable); no aparece si el baseline
    nunca predice esa clase.
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
            eje.plot(*punto, "o", color=COLORES[riesgo], markersize=7, markeredgecolor=FONDO)
            eje.annotate(
                "decisión del baseline",
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
        eje.set_facecolor(FONDO)
        eje.grid(color=CUADRICULA, linewidth=0.8)
        eje.set_axisbelow(True)
        eje.tick_params(colors=TEXTO_SECUNDARIO)
        eje.spines[["top", "right"]].set_visible(False)
        eje.spines[["left", "bottom"]].set_color(CUADRICULA)
    ejes[0].set_ylabel("Precisión", color=TEXTO_SECUNDARIO)
    figura.suptitle(titulo, x=0.01, ha="left", color=TEXTO)
    figura.tight_layout()
    return figura


def registrar_run(
    nombre: str,
    hiperparametros: dict,
    clase: pd.Series,
    probabilidades: pd.DataFrame,
    parametros: dict,
    etiquetas: dict,
) -> dict[str, float]:
    """Registra un baseline como un run de MLflow: parámetros, etiquetas, métricas y la gráfica."""
    with mlflow.start_run(run_name=nombre, tags=etiquetas):
        mlflow.log_params({**parametros, **hiperparametros})
        metricas = calcular_metricas(clase, probabilidades)
        mlflow.log_metrics(metricas)
        figura = graficar_curvas_pr(clase, probabilidades, f"P2 · {nombre} (conjunto de prueba)")
        mlflow.log_figure(figura, "curva_precision_recall.png")
    resumen = {
        metrica: round(valor, 3)
        for metrica, valor in metricas.items()
        if "proporcion" not in metrica
    }
    logger.info(f"{nombre}: {resumen}")
    return metricas


@app.command()
def main(ventanas_path: Path = P2_VENTANAS_PATH):
    tabla = pd.read_parquet(ventanas_path)
    tabla["conjunto"] = asignar_conjunto(tabla["dia"])
    entrenamiento = tabla[tabla["conjunto"] == "entrenamiento"]
    prueba = tabla[tabla["conjunto"] == "prueba"]
    logger.info(f"Ventanas: {len(entrenamiento):,} de entrenamiento y {len(prueba):,} de prueba")

    etiquetas = etiquetas_run("P2", __file__)
    parametros = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_inicio_validacion": FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": FECHA_INICIO_PRUEBA,
        "version_datos": version_datos(),
        "semilla": SEMILLA,
        "ventana": P2_VENTANA,
        "hora_reinicio": P2_HORA_REINICIO,
        "inventario_inicial": P2_INVENTARIO_INICIAL,
        "ventanas_entrenamiento": len(entrenamiento),
        "ventanas_prueba": len(prueba),
    }

    usar_experimento()

    registrar_run(
        "p2-baseline-clase-mayoritaria",
        {"modelo": "DummyClassifier", "strategy": "most_frequent"},
        prueba["clase"],
        baseline_clase_mayoritaria(entrenamiento, prueba),
        parametros,
        etiquetas,
    )
    registrar_run(
        f"p2-baseline-frecuencia-{P2_SEMANAS_HISTORIA}-semanas",
        {
            "modelo": "frecuencia_semanas_anteriores",
            "semanas": P2_SEMANAS_HISTORIA,
            "desempate": " > ".join(CLASES_P2),
        },
        prueba["clase"],
        baseline_frecuencia(prueba),
        parametros,
        etiquetas,
    )
    logger.success(
        f"Runs registrados en el experimento {EXPERIMENTO_MLFLOW} ({MLFLOW_TRACKING_URI})"
    )


if __name__ == "__main__":
    app()
