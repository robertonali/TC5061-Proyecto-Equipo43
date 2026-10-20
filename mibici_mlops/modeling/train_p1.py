"""Entrena y evalúa P1 (demanda de viajes por estación y día) y lo registra en MLflow.

Compara el baseline (mismo día de la semana pasada) contra un gradient boosting. Los
hiperparámetros se eligen con validación y la prueba se evalúa una sola vez, al final.

Antes de correrlo, el servidor de MLflow debe estar levantado (ver la sección de MLflow del
README):
    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination mlruns --port 5000 --workers 1

Uso:
    python -m mibici_mlops.modeling.train_p1

Los runs se ven en http://127.0.0.1:5000.
"""

from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import typer

from mibici_mlops.config import (
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    FIGURES_DIR,
    P1_FEATURES_PATH,
    P1_MODEL_PATH,
    SEMILLA,
)
from mibici_mlops.features import COLUMNAS_P1
from mibici_mlops.modeling.registro import etiquetas_run, usar_experimento, version_datos

app = typer.Typer()

# Candidatos del gradient boosting. El conteo de viajes es no negativo, por lo que se prueba la
# pérdida de Poisson además del error cuadrático.
CANDIDATOS = [
    {"loss": "poisson", "learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31},
    {"loss": "poisson", "learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 63},
    {"loss": "squared_error", "learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31},
]


def metricas(real: pd.Series, prediccion: np.ndarray, prefijo: str) -> dict[str, float]:
    """MAE y RMSE en número de viajes por estación-día."""
    return {
        f"{prefijo}_mae": float(mean_absolute_error(real, prediccion)),
        f"{prefijo}_rmse": float(root_mean_squared_error(real, prediccion)),
    }


DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def error_por_dia_semana(
    prueba: pd.DataFrame, predicciones: dict[str, np.ndarray]
) -> pd.DataFrame:
    """MAE por día de la semana (0 = lunes) para cada modelo."""
    filas = {
        nombre: (prueba["viajes"] - pred).abs().groupby(prueba["dia_semana"]).mean()
        for nombre, pred in predicciones.items()
    }
    return pd.DataFrame(filas).rename_axis("dia_semana").reset_index()


def error_por_nivel_uso(
    entrenamiento: pd.DataFrame, prueba: pd.DataFrame, predicciones: dict[str, np.ndarray]
) -> pd.DataFrame:
    """MAE por nivel de uso de la estación (terciles de la demanda media en entrenamiento).

    Los terciles se calculan solo con el entrenamiento. Las estaciones sin historial de
    entrenamiento se reportan aparte, como "sin historial".
    """
    media = entrenamiento.groupby("estacion_id")["viajes"].mean()
    cortes = media.quantile([1 / 3, 2 / 3]).to_numpy()
    nivel = pd.cut(
        media, [-np.inf, cortes[0], cortes[1], np.inf], labels=["bajo", "medio", "alto"]
    )
    nivel_fila = prueba["estacion_id"].map(nivel).astype(object).fillna("sin historial")
    resultado = {"viajes_medios": prueba["viajes"].groupby(nivel_fila).mean()}
    for nombre, pred in predicciones.items():
        resultado[nombre] = (prueba["viajes"] - pred).abs().groupby(nivel_fila).mean()
    tabla = pd.DataFrame(resultado).rename_axis("nivel_uso").reset_index()
    tabla["filas"] = tabla["nivel_uso"].map(nivel_fila.value_counts())
    return tabla


def graficar(
    prueba: pd.DataFrame,
    predicciones: dict[str, np.ndarray],
    por_dia: pd.DataFrame,
    por_nivel: pd.DataFrame,
) -> list[Path]:
    """Genera las gráficas de P1 en reports/figures y regresa sus rutas."""
    plt.switch_backend("Agg")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rutas = []

    # 1. Demanda real y predicha, total por día en el periodo de prueba.
    diario = prueba[["fecha", "viajes"]].copy()
    for nombre, pred in predicciones.items():
        diario[nombre] = pred
    diario = diario.groupby("fecha").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(diario.index, diario["viajes"], color="black", label="Real")
    for nombre in predicciones:
        ax.plot(diario.index, diario[nombre], alpha=0.8, label=nombre)
    ax.set(title="P1: viajes por día en prueba (suma de las estaciones)", ylabel="Viajes")
    ax.legend()
    fig.autofmt_xdate()
    rutas.append(FIGURES_DIR / "p1_real_vs_predicho.png")
    fig.savefig(rutas[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Error por día de la semana.
    fig, ax = plt.subplots(figsize=(8, 4))
    for nombre in predicciones:
        ax.plot(por_dia["dia_semana"], por_dia[nombre], marker="o", label=nombre)
    ax.set_xticks(range(7), DIAS)
    ax.set(title="P1: MAE por día de la semana (prueba)", ylabel="MAE (viajes)")
    ax.legend()
    rutas.append(FIGURES_DIR / "p1_error_por_dia_semana.png")
    fig.savefig(rutas[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Error por nivel de uso de la estación.
    fig, ax = plt.subplots(figsize=(8, 4))
    posiciones = np.arange(len(por_nivel))
    ancho = 0.8 / len(predicciones)
    for i, nombre in enumerate(predicciones):
        ax.bar(posiciones + i * ancho, por_nivel[nombre], ancho, label=nombre)
    ax.set_xticks(posiciones + ancho * (len(predicciones) - 1) / 2, por_nivel["nivel_uso"])
    ax.set(title="P1: MAE por nivel de uso de la estación (prueba)", ylabel="MAE (viajes)")
    ax.legend()
    rutas.append(FIGURES_DIR / "p1_error_por_nivel_uso.png")
    fig.savefig(rutas[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rutas


def parametros_comunes(datos: dict[str, pd.DataFrame]) -> dict[str, str | int]:
    return {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_inicio_validacion": FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": FECHA_INICIO_PRUEBA,
        "version_datos": version_datos(),
        "semilla": SEMILLA,
        "n_filas_entrenamiento": len(datos["entrenamiento"]),
        "n_filas_validacion": len(datos["validacion"]),
        "n_filas_prueba": len(datos["prueba"]),
    }


@app.command()
def main(
    features_path: Path = P1_FEATURES_PATH,
    model_path: Path = P1_MODEL_PATH,
    muestra_entrenamiento: float = typer.Option(
        1.0, help="Fracción de filas de entrenamiento a usar (para pruebas rápidas)."
    ),
):
    tabla = pd.read_parquet(features_path)
    datos = {nombre: grupo for nombre, grupo in tabla.groupby("conjunto", observed=True)}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        logger.info(f"{nombre}: {len(datos[nombre]):,} filas")
    entrenamiento, validacion, prueba = (
        datos[n] for n in ("entrenamiento", "validacion", "prueba")
    )
    if muestra_entrenamiento < 1.0:
        entrenamiento = entrenamiento.sample(frac=muestra_entrenamiento, random_state=SEMILLA)
        logger.warning(
            f"Entrenando con {len(entrenamiento):,} filas ({muestra_entrenamiento:.0%})"
        )

    usar_experimento()
    etiquetas = etiquetas_run("P1", __file__)
    comunes = parametros_comunes(datos)

    # ---- Baseline: el mismo día de la semana pasada (lag_7d) ----
    with mlflow.start_run(run_name="p1-baseline-semana-previa", tags=etiquetas):
        mlflow.log_params({**comunes, "modelo": "naive_estacional_semana_previa"})
        pred_base = {c: datos[c]["lag_7d"].to_numpy() for c in ("validacion", "prueba")}
        resultados_base = {
            **metricas(validacion["viajes"], pred_base["validacion"], "val"),
            **metricas(prueba["viajes"], pred_base["prueba"], "test"),
        }
        mlflow.log_metrics(resultados_base)
    logger.info(f"Baseline: {resultados_base}")

    # ---- Candidato: gradient boosting, hiperparámetros elegidos con validación ----
    x_entrenamiento = entrenamiento[COLUMNAS_P1]
    mejor, busqueda = None, []
    for params in CANDIDATOS:
        modelo = HistGradientBoostingRegressor(
            **params, early_stopping=False, random_state=SEMILLA
        ).fit(x_entrenamiento, entrenamiento["viajes"])
        pred_val = np.clip(modelo.predict(validacion[COLUMNAS_P1]), 0, None)
        val = metricas(validacion["viajes"], pred_val, "val")
        busqueda.append({**params, **val})
        logger.info(f"{params} -> {val}")
        if mejor is None or val["val_mae"] < mejor["val"]["val_mae"]:
            mejor = {"modelo": modelo, "params": params, "val": val}

    modelo = mejor["modelo"]
    # La prueba se evalúa una sola vez, con el modelo ya elegido.
    pred_prueba = np.clip(modelo.predict(prueba[COLUMNAS_P1]), 0, None)
    resultados = {**mejor["val"], **metricas(prueba["viajes"], pred_prueba, "test")}

    predicciones = {"baseline": pred_base["prueba"], "gradient_boosting": pred_prueba}
    por_dia = error_por_dia_semana(prueba, predicciones)
    por_nivel = error_por_nivel_uso(entrenamiento, prueba, predicciones)
    figuras = graficar(prueba, predicciones, por_dia, por_nivel)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rutas_tablas = {
        FIGURES_DIR / "p1_busqueda_hiperparametros.csv": pd.DataFrame(busqueda),
        FIGURES_DIR / "p1_error_por_dia_semana.csv": por_dia,
        FIGURES_DIR / "p1_error_por_nivel_uso.csv": por_nivel,
    }
    for ruta, df in rutas_tablas.items():
        df.to_csv(ruta, index=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, model_path)

    with mlflow.start_run(run_name="p1-gradient-boosting", tags=etiquetas):
        mlflow.log_params(
            {**comunes, "modelo": "HistGradientBoostingRegressor", **mejor["params"]}
        )
        mlflow.log_metrics(resultados)
        for ruta in [*figuras, *rutas_tablas, model_path]:
            mlflow.log_artifact(str(ruta))
    logger.success(f"Gradient boosting: {resultados}")
    logger.info(f"Baseline:          {resultados_base}")


if __name__ == "__main__":
    app()
