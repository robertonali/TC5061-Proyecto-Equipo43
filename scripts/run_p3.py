"""Ejecuta P3 de punta a punta: candidatos, campeón en MLflow y servidor de inferencia.

El servidor de tracking de MLflow debe estar activo en MLFLOW_TRACKING_URI antes de ejecutar
este script. Por defecto, al final se queda sirviendo el alias `champion` en el puerto 5001.
Usa `--no-serve` para terminar después de entrenar, registrar y guardar las asignaciones.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import mlflow
import pandas as pd
from loguru import logger
from mlflow import MlflowClient

from mibici_mlops.config import (
    ESTACIONES_PATH,
    MLFLOW_TRACKING_URI,
    P3_CLUSTERS_PATH,
    P3_MODELO_REGISTRADO,
    P3_MODEL_PATH,
    P3_PERFILES_PATH,
    VIAJES_PATH,
)
from mibici_mlops.dataset import main as preparar_datos
from mibici_mlops.features import p3 as generar_perfiles
from mibici_mlops.modeling.registrar_p3 import (
    main as registrar_campeon,
    modelo_del_run,
)
from mibici_mlops.modeling.registro import verificar_servidor
from mibici_mlops.modeling.train_p3 import MODELOS, p3train as entrenar
from mibici_mlops.plots import p3 as generar_plots


def entrenar_candidatos() -> list[str]:
    """Entrena los modelos configurados y devuelve sus run IDs, ordenados por ARI."""
    cliente = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    resultados = []

    for nombre_modelo in MODELOS:
        logger.info("Entrenando candidato P3: {}", nombre_modelo)
        run_id = entrenar(modelo=nombre_modelo)
        run = cliente.get_run(run_id)
        ari = run.data.metrics.get("ari_estabilidad")
        if ari is None:
            raise RuntimeError(f"El run {run_id} no registró ari_estabilidad.")
        resultados.append((float(ari), run_id))
        logger.info("{}: ari_estabilidad={:.4f}; run_id={}", nombre_modelo, ari, run_id)

    resultados.sort(key=lambda resultado: resultado[0], reverse=True)
    if not resultados:
        raise RuntimeError("No se configuraron candidatos de P3 para entrenar.")
    return [run_id for _, run_id in resultados]


def guardar_salida_del_campeon(run_id: str) -> None:
    """Actualiza los archivos locales con el modelo y las asignaciones del campeón."""
    cliente = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    modelo_uri = modelo_del_run(cliente.get_run(run_id))
    modelo = mlflow.sklearn.load_model(modelo_uri)

    perfiles = pd.read_parquet(P3_PERFILES_PATH)
    columnas = [c for c in perfiles.columns if c.startswith(("sal_", "lle_"))]
    partes = []

    for conjunto in ("entrenamiento", "validacion", "prueba"):
        filas = perfiles.loc[perfiles["conjunto"].eq(conjunto)]
        if filas.empty:
            continue
        etiquetas = modelo.predict(filas[columnas].astype("float64"))
        partes.append(
            pd.DataFrame(
                {
                    "estacion": filas["estacion"].to_numpy(),
                    "conjunto": conjunto,
                    "cluster": etiquetas,
                }
            )
        )

    estaciones = (
        pd.read_csv(ESTACIONES_PATH)[["id", "latitude", "longitude"]]
        .rename(
            columns={
                "id": "estacion",
                "latitude": "estacion_lat",
                "longitude": "estacion_lon",
            }
        )
    )
    asignaciones = pd.concat(partes, ignore_index=True).merge(
        estaciones,
        on="estacion",
        how="left",
        validate="many_to_one",
    )

    P3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    P3_CLUSTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, P3_MODEL_PATH)
    asignaciones.to_parquet(P3_CLUSTERS_PATH, index=False, compression="zstd")
    generar_plots(run_id=run_id)

    logger.success("Artefactos locales actualizados con el campeón {}", run_id)


def servir_campeon(port: int) -> None:
    """Arranca el servidor HTTP de MLflow para el alias champion."""
    uri = f"models:/{P3_MODELO_REGISTRADO}@champion"
    logger.success("Sirviendo {} en http://127.0.0.1:{}/", uri, port)
    logger.info("Endpoint de predicción: http://127.0.0.1:{}/invocations", port)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "serve",
            "--model-uri",
            uri,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--env-manager",
            "local",
        ],
        cwd=ROOT,
        check=True,
    )


def main(port: int = 5001, serve: bool = True) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    verificar_servidor()

    print("\n[PIPELINE] Preparando data/interim/viajes.parquet")
    if not VIAJES_PATH.exists():
        print("\n[PIPELINE] Preparando datos desde los CSV locales")
        preparar_datos(descargar_csv=False)
    else:
        print(f"\n[PIPELINE] Reutilizando {VIAJES_PATH}")

    print("\n[PIPELINE] Generando perfiles P3")
    generar_perfiles()

    print("\n[PIPELINE] Entrenando candidatos y registrando runs en MLflow")
    run_ids = entrenar_candidatos()
    champion_id = run_ids[0]
    challenger_id = run_ids[1] if len(run_ids) > 1 else None

    print(f"\n[PIPELINE] Champion: {champion_id}")
    if challenger_id:
        print(f"[PIPELINE] Challenger: {challenger_id}")
    registrar_campeon(champion=champion_id, challenger=challenger_id)

    print("\n[PIPELINE] Guardando predicciones y artefactos del champion")
    guardar_salida_del_campeon(champion_id)

    print("\n[PIPELINE] Entrenamiento y registro P3 completados")
    if serve:
        servir_campeon(port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el flujo completo de P3 con MLflow.")
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Puerto del servidor HTTP del modelo (default: 5001).",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Entrena y registra el champion, pero no arranca el servidor HTTP.",
    )
    argumentos = parser.parse_args()
    main(port=argumentos.port, serve=not argumentos.no_serve)
