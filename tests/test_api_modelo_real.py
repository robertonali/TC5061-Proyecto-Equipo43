"""Prueba de integración: la API con el modelo exportado del registry, sobre filas reales.

La API sirve los modelos que estén exportados en models/servicio/p1/ y models/servicio/p3/
(`python -m mibici_mlops.api.modelo exportar --pregunta p1` y `--pregunta p3`). Las pruebas de
cada pregunta se saltan si falta su modelo o sus datos (data/processed/p1_estacion_dia.parquet y
data/processed/p3_perfiles.parquet).
"""

import json

from fastapi.testclient import TestClient
import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici_mlops.api.app import app
from mibici_mlops.config import (
    P1_FEATURES_PATH,
    P3_PERFILES_PATH,
    SERVICIO_MODELO_DIR,
    SERVICIO_P3_MODELO_DIR,
)
from mibici_mlops.features import COLUMNAS_P1, COLUMNAS_P3

N_FILAS = 400


@pytest.fixture(scope="module")
def cliente():
    if not any((c / "MLmodel").exists() for c in (SERVICIO_MODELO_DIR, SERVICIO_P3_MODELO_DIR)):
        pytest.skip("No hay ningún modelo exportado en models/servicio/")
    # Con `with`, TestClient corre el arranque de la API, que carga los modelos como en producción.
    with pytest.MonkeyPatch.context() as mp:
        # La copia empaquetada, no el registry en vivo.
        mp.delenv("MODELO_URI", raising=False)
        mp.delenv("MODELO_URI_P3", raising=False)
        with TestClient(app) as cliente:
            yield cliente


@pytest.fixture(scope="module")
def filas_reales():
    if not (SERVICIO_MODELO_DIR / "MLmodel").exists():
        pytest.skip(f"Falta el modelo exportado en {SERVICIO_MODELO_DIR}")
    if not P1_FEATURES_PATH.exists():
        pytest.skip("Faltan los datos de P1 (data/processed/p1_estacion_dia.parquet)")
    return pd.read_parquet(P1_FEATURES_PATH, columns=["estacion_id", *COLUMNAS_P1]).tail(N_FILAS)


def test_sirve_la_version_exportada(cliente, filas_reales):
    info = cliente.get("/modelo").json()
    assert info["origen"] == "imagen"
    assert info["nombre"] == "mibici-equipo43-p1-demanda"
    salud = cliente.get("/salud").json()
    assert salud["estado"] == "ok"
    assert salud["version_modelo"] == info["version"]


def test_predicciones_iguales_al_modelo_cargado_directamente(cliente, filas_reales):
    filas = json.loads(filas_reales.to_json(orient="records"))
    respuesta = cliente.post("/prediccion", json={"filas": filas})
    assert respuesta.status_code == 200
    predicciones = respuesta.json()["predicciones"]

    directo = mlflow.pyfunc.load_model(str(SERVICIO_MODELO_DIR)).predict(filas_reales[COLUMNAS_P1])
    api = np.array([p["viajes"] for p in predicciones])
    # La API redondea a 2 decimales y recorta los negativos a 0.
    np.testing.assert_allclose(api, np.clip(directo, 0, None), atol=0.005 + 1e-9)
    assert [p["estacion_id"] for p in predicciones] == filas_reales["estacion_id"].tolist()


@pytest.fixture(scope="module")
def perfiles_reales():
    if not (SERVICIO_P3_MODELO_DIR / "MLmodel").exists():
        pytest.skip(f"Falta el modelo exportado en {SERVICIO_P3_MODELO_DIR}")
    if not P3_PERFILES_PATH.exists():
        pytest.skip("Faltan los datos de P3 (data/processed/p3_perfiles.parquet)")
    perfiles = pd.read_parquet(P3_PERFILES_PATH, columns=["estacion", "conjunto", *COLUMNAS_P3])
    return perfiles[perfiles["conjunto"] == "validacion"].head(N_FILAS)


def test_sirve_la_version_exportada_de_p3(cliente, perfiles_reales):
    info = cliente.get("/clusters/modelo").json()
    assert info["origen"] == "imagen"
    assert info["nombre"] == "mibici-equipo43-p3-clusters"
    assert cliente.get("/salud").json()["version_modelo_p3"] == info["version"]


def test_clusters_iguales_al_modelo_cargado_directamente(cliente, perfiles_reales):
    filas = json.loads(
        perfiles_reales.rename(columns={"estacion": "estacion_id"})
        .drop(columns="conjunto")
        .to_json(orient="records")
    )
    respuesta = cliente.post("/clusters", json={"filas": filas})
    assert respuesta.status_code == 200
    predicciones = respuesta.json()["predicciones"]
    directo = mlflow.pyfunc.load_model(str(SERVICIO_P3_MODELO_DIR)).predict(
        perfiles_reales[COLUMNAS_P3].astype("float64")
    )
    assert [p["cluster"] for p in predicciones] == [int(g) for g in directo]
    assert [p["estacion_id"] for p in predicciones] == perfiles_reales["estacion"].tolist()
