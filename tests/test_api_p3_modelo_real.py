"""Prueba de integración de P3: la API con el modelo exportado del registry, sobre perfiles reales.

Se salta si falta alguno de los modelos exportados (la API carga P1, P2 y P3 al arrancar; ver
`python -m mibici_mlops.api.modelo exportar`) o si no están los perfiles de P3
(data/processed/p3_perfiles.parquet).
"""

import json

from fastapi.testclient import TestClient
import mlflow
import pandas as pd
import pytest

from mibici_mlops.api.app import app
from mibici_mlops.config import (
    P2_SERVICIO_MODELO_DIR,
    P3_PERFILES_PATH,
    P3_SERVICIO_MODELO_DIR,
    SERVICIO_MODELO_DIR,
)
from mibici_mlops.features import COLUMNAS_P3

N_FILAS = 400

pytestmark = pytest.mark.skipif(
    not all(
        (carpeta / "MLmodel").exists()
        for carpeta in (SERVICIO_MODELO_DIR, P2_SERVICIO_MODELO_DIR, P3_SERVICIO_MODELO_DIR)
    )
    or not P3_PERFILES_PATH.exists(),
    reason="Faltan los modelos exportados en models/servicio/ o los perfiles de P3",
)


@pytest.fixture(scope="module")
def cliente():
    # Con `with`, TestClient corre el arranque de la API, que carga los modelos como en producción.
    with pytest.MonkeyPatch.context() as mp:
        for variable in ("MODELO_URI", "MODELO_URI_P2", "MODELO_URI_P3"):
            mp.delenv(variable, raising=False)  # las copias empaquetadas, no el registry en vivo
        with TestClient(app) as cliente:
            yield cliente


@pytest.fixture(scope="module")
def perfiles_reales():
    perfiles = pd.read_parquet(P3_PERFILES_PATH, columns=["estacion", "conjunto", *COLUMNAS_P3])
    return perfiles[perfiles["conjunto"] == "validacion"].head(N_FILAS)


def test_sirve_la_version_exportada_de_p3(cliente):
    info = cliente.get("/p3/modelo").json()
    assert info["origen"] == "imagen"
    assert info["nombre"] == "mibici-equipo43-p3-clusters"
    assert cliente.get("/salud").json()["modelos"]["p3"] == info["version"]


def test_grupos_iguales_al_modelo_cargado_directamente(cliente, perfiles_reales):
    filas = json.loads(
        perfiles_reales.rename(columns={"estacion": "estacion_id"})
        .drop(columns="conjunto")
        .to_json(orient="records")
    )
    respuesta = cliente.post("/p3/prediccion", json={"filas": filas})
    assert respuesta.status_code == 200
    predicciones = respuesta.json()["predicciones"]
    directo = mlflow.pyfunc.load_model(str(P3_SERVICIO_MODELO_DIR)).predict(
        perfiles_reales[COLUMNAS_P3].astype("float64")
    )
    assert [p["cluster"] for p in predicciones] == [int(g) for g in directo]
    assert [p["estacion_id"] for p in predicciones] == perfiles_reales["estacion"].tolist()
