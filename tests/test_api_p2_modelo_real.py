"""Prueba de integración de P2: la API con el modelo exportado del registry, sobre ventanas reales.

Se salta si falta alguno de los dos modelos exportados (la API carga P1 y P2 al arrancar; ver
`python -m mibici_mlops.api.modelo exportar --question p2`) o la tabla de P2
(data/processed/p2_ventanas.parquet).
"""

import json

from fastapi.testclient import TestClient
import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici_mlops.api import p2
from mibici_mlops.api.app import app
from mibici_mlops.config import P2_SERVICIO_MODELO_DIR, P2_VENTANAS_PATH, SERVICIO_MODELO_DIR
from mibici_mlops.features import CLASES_P2, VARIABLES_P2

N_FILAS = 400

pytestmark = pytest.mark.skipif(
    not (SERVICIO_MODELO_DIR / "MLmodel").exists()
    or not (P2_SERVICIO_MODELO_DIR / "MLmodel").exists()
    or not P2_VENTANAS_PATH.exists(),
    reason="Faltan los modelos exportados en models/servicio/p1/ y p2/ o la tabla de P2",
)


@pytest.fixture(scope="module")
def cliente():
    # Con `with`, TestClient corre el arranque de la API, que carga los modelos como en producción.
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("MODELO_URI", raising=False)  # la copia empaquetada, no el registry en vivo
        mp.delenv("MODELO_URI_P2", raising=False)
        with TestClient(app) as cliente:
            yield cliente


@pytest.fixture(scope="module")
def ventanas_reales():
    columnas = ["estacion", *VARIABLES_P2]
    return pd.read_parquet(P2_VENTANAS_PATH, columns=columnas).tail(N_FILAS).reset_index(drop=True)


def test_sirve_la_version_exportada(cliente):
    info = cliente.get("/p2/modelo").json()
    assert info["origen"] == "imagen"
    assert info["nombre"] == "mibici-equipo43-p2-riesgo"
    salud = cliente.get("/salud").json()
    assert salud["estado"] == "ok"
    assert salud["modelos"]["p2"] == info["version"]


def test_probabilidades_iguales_al_modelo_cargado_directamente(cliente, ventanas_reales):
    filas = json.loads(
        ventanas_reales.rename(columns={"estacion": "estacion_id"}).to_json(orient="records")
    )
    respuesta = cliente.post("/p2/prediccion", json={"filas": filas})
    assert respuesta.status_code == 200, respuesta.text
    predicciones = respuesta.json()["predicciones"]

    directo = mlflow.sklearn.load_model(str(P2_SERVICIO_MODELO_DIR))
    esperadas = pd.DataFrame(
        directo.predict_proba(ventanas_reales[VARIABLES_P2]), columns=directo.classes_
    )[CLASES_P2]
    api = np.array([[p["probabilidades"][c] for c in CLASES_P2] for p in predicciones])
    # La API redondea a 3 decimales.
    np.testing.assert_allclose(api, esperadas.to_numpy(), atol=0.0005 + 1e-9)
    assert [p["clase"] for p in predicciones] == p2.clase_predicha(esperadas)
    assert [p["estacion_id"] for p in predicciones] == ventanas_reales["estacion"].tolist()
    assert [p["hora"] for p in predicciones] == ventanas_reales["hora"].tolist()
