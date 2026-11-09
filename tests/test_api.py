"""Pruebas de la API: validación de entradas, formato de la respuesta y recorte de negativos.

Usan un modelo falso en lugar del exportado del registry, así que corren sin servidor de MLflow
ni models/servicio/p1/.
"""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici_mlops.api import modelo as modulo_modelo
from mibici_mlops.api import p2
from mibici_mlops.api.app import app, obtener_modelo
from mibici_mlops.api.modelo import metadatos, separar_uri
from mibici_mlops.features import COLUMNAS_P1

INFO = {
    "nombre": "mibici-equipo43-p1-demanda",
    "alias": "champion",
    "version": "3",
    "run_id": "abc123",
    "origen": "imagen",
}


class ModeloFalso:
    """Predice media_7d - 10, para que haya negativos que recortar. Recuerda lo que recibió."""

    metadata = None

    def predict(self, tabla: pd.DataFrame) -> np.ndarray:
        self.recibido = tabla
        return tabla["media_7d"].to_numpy() - 10


@pytest.fixture
def modelo():
    return ModeloFalso()


@pytest.fixture
def cliente(modelo):
    # Sin `with`, TestClient no corre el arranque, que cargaría los modelos reales. /salud también
    # pregunta por el modelo de P2 (sus pruebas están en test_api_p2.py); aquí basta su versión.
    app.dependency_overrides[obtener_modelo] = lambda: (modelo, INFO)
    app.dependency_overrides[p2.obtener_modelo_p2] = lambda: (None, {"version": "1"})
    yield TestClient(app)
    app.dependency_overrides.clear()


def fila(**cambios) -> dict:
    base = {
        "estacion_id": 2,
        "dia_semana": 2,
        "mes": 10,
        "es_fin_de_semana": 0,
        "es_feriado": 0,
        "capacidad": 23,
        "lat": 20.67,
        "lon": -103.37,
        "lag_7d": 41,
        "media_7d": 37.4,
    }
    return {**base, **cambios}


def test_salud_y_modelo(cliente):
    assert cliente.get("/salud").json() == {"estado": "ok", "modelos": {"p1": "3", "p2": "1"}}
    assert cliente.get("/modelo").json() == INFO


def test_prediccion_devuelve_una_por_fila_y_recorta_negativos(cliente, modelo):
    filas = [fila(estacion_id=1, media_7d=37.4), fila(estacion_id=7, media_7d=4.0)]
    respuesta = cliente.post("/prediccion", json={"filas": filas})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo"] == INFO
    assert cuerpo["predicciones"] == [
        {"estacion_id": 1, "viajes": 27.4},
        {"estacion_id": 7, "viajes": 0.0},
    ]
    # El modelo recibe exactamente sus columnas, en orden y sin estacion_id.
    assert list(modelo.recibido.columns) == COLUMNAS_P1


def test_estacion_id_es_opcional(cliente):
    sin_id = {k: v for k, v in fila().items() if k != "estacion_id"}
    respuesta = cliente.post("/prediccion", json={"filas": [sin_id]})
    assert respuesta.status_code == 200
    assert respuesta.json()["predicciones"][0]["estacion_id"] is None


@pytest.mark.parametrize(
    "cambios",
    [
        {"dia_semana": 7},
        {"mes": 0},
        {"es_feriado": 2},
        {"capacidad": 0},
        {"lat": 19.4},  # fuera de Guadalajara
        {"lag_7d": -1},
        {"media_7d": "muchos"},
        {"dia_semana": 5, "es_fin_de_semana": 0},  # sábado marcado como entre semana
        {"columna_extra": 1},
    ],
)
def test_rechaza_filas_invalidas(cliente, cambios):
    respuesta = cliente.post("/prediccion", json={"filas": [fila(**cambios)]})
    assert respuesta.status_code == 422


@pytest.mark.parametrize("valor", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("campo", ["lag_7d", "media_7d", "capacidad"])
def test_rechaza_valores_no_finitos_con_422(cliente, campo, valor):
    # FastAPI acepta NaN e Infinity en el JSON; deben rechazarse con 422, no aceptarse ni dar 500.
    # json.dumps los escribe tal cual (NaN, Infinity), como los mandaría un cliente.
    cuerpo = json.dumps({"filas": [fila(**{campo: float(valor)})]})
    assert valor in cuerpo
    respuesta = cliente.post(
        "/prediccion", content=cuerpo, headers={"Content-Type": "application/json"}
    )
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "filas", 0, campo]


def test_rechaza_filas_incompletas_o_vacias(cliente):
    incompleta = {k: v for k, v in fila().items() if k != "lag_7d"}
    assert cliente.post("/prediccion", json={"filas": [incompleta]}).status_code == 422
    assert cliente.post("/prediccion", json={"filas": []}).status_code == 422


@pytest.mark.parametrize(
    ("uri", "esperado"),
    [
        (
            "models:/mibici-equipo43-p1-demanda@champion",
            ("mibici-equipo43-p1-demanda", "champion", None),
        ),
        ("models:/mibici-equipo43-p1-demanda/3", ("mibici-equipo43-p1-demanda", None, "3")),
    ],
)
def test_separar_uri_por_alias_o_version(uri, esperado):
    assert separar_uri(uri) == esperado


@pytest.mark.parametrize(
    "uri",
    [
        "mibici-equipo43-p1-demanda@champion",  # sin models:/
        "models:/",
        "models:/mibici-equipo43-p1-demanda",  # sin alias ni versión
        "models:/mibici-equipo43-p1-demanda/latest",  # la versión debe ser un número
        "models:/mibici-equipo43-p1-demanda/3@champion",
        "models:/@champion",
        "runs:/abc123/modelo",
    ],
)
def test_separar_uri_rechaza_formatos_invalidos(uri):
    with pytest.raises(ValueError, match="Se esperaba"):
        separar_uri(uri)


class ClienteFalso:
    """MlflowClient con una sola versión registrada: la 3, con alias champion."""

    version = SimpleNamespace(version="3", run_id="abc123")

    def get_model_version_by_alias(self, nombre, alias):
        assert alias == "champion"
        return self.version

    def get_model_version(self, nombre, version):
        assert version == "3"
        return self.version


@pytest.mark.parametrize(
    ("uri", "alias"),
    [
        ("models:/mibici-equipo43-p1-demanda@champion", "champion"),
        ("models:/mibici-equipo43-p1-demanda/3", None),
    ],
)
def test_metadatos_por_alias_o_version(monkeypatch, uri, alias):
    monkeypatch.setattr(modulo_modelo, "MlflowClient", ClienteFalso)
    assert metadatos(uri) == {
        "nombre": "mibici-equipo43-p1-demanda",
        "alias": alias,
        "version": "3",
        "run_id": "abc123",
    }
