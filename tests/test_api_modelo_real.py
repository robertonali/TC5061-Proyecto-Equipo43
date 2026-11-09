"""Prueba de integración: la API con el modelo exportado del registry, sobre filas reales.

Se salta si falta alguno de los modelos exportados (la API carga P1, P2 y P3 al arrancar; ver
`python -m mibici_mlops.api.modelo exportar`) o si no están los datos de P1
(data/processed/p1_estacion_dia.parquet).
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
    P2_SERVICIO_MODELO_DIR,
    P3_SERVICIO_MODELO_DIR,
    SERVICIO_MODELO_DIR,
)
from mibici_mlops.features import COLUMNAS_P1

N_FILAS = 400

pytestmark = pytest.mark.skipif(
    not (SERVICIO_MODELO_DIR / "MLmodel").exists()
    or not (P2_SERVICIO_MODELO_DIR / "MLmodel").exists()
    or not (P3_SERVICIO_MODELO_DIR / "MLmodel").exists()
    or not P1_FEATURES_PATH.exists(),
    reason="Faltan los modelos exportados en models/servicio/ (p1, p2 y p3) o los datos de P1",
)


@pytest.fixture(scope="module")
def cliente():
    # Con `with`, TestClient corre el arranque de la API, que carga el modelo como en producción.
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("MODELO_URI", raising=False)  # la copia empaquetada, no el registry en vivo
        mp.delenv("MODELO_URI_P2", raising=False)
        mp.delenv("MODELO_URI_P3", raising=False)
        with TestClient(app) as cliente:
            yield cliente


@pytest.fixture(scope="module")
def filas_reales():
    return pd.read_parquet(P1_FEATURES_PATH, columns=["estacion_id", *COLUMNAS_P1]).tail(N_FILAS)


def test_sirve_la_version_exportada(cliente):
    info = cliente.get("/modelo").json()
    assert info["origen"] == "imagen"
    assert info["nombre"] == "mibici-equipo43-p1-demanda"
    assert cliente.get("/salud").json()["modelos"]["p1"] == info["version"]


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
