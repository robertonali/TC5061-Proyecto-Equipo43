"""Pruebas de la API de P3 (grupo de una estación según su perfil de uso).

Usan un modelo falso en lugar del exportado del registry, así que corren sin servidor de MLflow
ni models/servicio/p3/.
"""

import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici_mlops.api import p2, p3
from mibici_mlops.api.app import app, obtener_modelo
from mibici_mlops.api.p3 import PerfilEstacion
from mibici_mlops.features import COLUMNAS_P3, columnas_perfil

INFO_P3 = {
    "nombre": "mibici-equipo43-p3-clusters",
    "alias": "champion",
    "version": "2",
    "run_id": "def456",
    "origen": "imagen",
}
INFO_P1 = {**INFO_P3, "nombre": "mibici-equipo43-p1-demanda", "run_id": "abc123"}


class ModeloFalso:
    """Grupo 1 si la estación sale sobre todo el lunes por la mañana; si no, grupo 0."""

    def predict(self, tabla: pd.DataFrame) -> np.ndarray:
        self.recibido = tabla
        return (tabla["sal_lun_manana"].to_numpy() > 0.5).astype(np.int32)


class ModeloP1Falso:
    metadata = None

    def predict(self, tabla: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(tabla))


@pytest.fixture
def modelo():
    return ModeloFalso()


@pytest.fixture
def cliente(modelo):
    # Sin `with`, TestClient no corre el arranque, que cargaría los modelos reales.
    app.dependency_overrides[p3.obtener_modelo_p3] = lambda: (modelo, INFO_P3)
    app.dependency_overrides[obtener_modelo] = lambda: (ModeloP1Falso(), INFO_P1)
    app.dependency_overrides[p2.obtener_modelo_p2] = lambda: (None, {"version": "1"})
    yield TestClient(app)
    app.dependency_overrides.clear()


def perfil(salida_lunes_manana: float = 1 / 28, **cambios) -> dict:
    """Un perfil válido: salidas y llegadas repartidas por igual, salvo sal_lun_manana.

    El resto de las salidas se reparte por igual para que el bloque siga sumando 1.
    """
    otras = (1 - salida_lunes_manana) / 27
    fila = {c: otras for c in columnas_perfil("sal")}
    fila["sal_lun_manana"] = salida_lunes_manana
    fila.update({c: 1 / 28 for c in columnas_perfil("lle")})
    return {"estacion_id": 7, **fila, **cambios}


def test_el_perfil_tiene_56_columnas_en_orden():
    assert len(COLUMNAS_P3) == 56
    assert COLUMNAS_P3[0] == "sal_lun_madrugada"
    assert COLUMNAS_P3[27] == "sal_dom_noche"
    assert COLUMNAS_P3[28] == "lle_lun_madrugada"
    assert set(PerfilEstacion.model_fields) == {"estacion_id", *COLUMNAS_P3}


def test_clusters_devuelve_uno_por_perfil(cliente, modelo):
    filas = [
        perfil(salida_lunes_manana=0.9, estacion_id=1),
        perfil(salida_lunes_manana=1 / 28, estacion_id=2),
    ]
    respuesta = cliente.post("/p3/prediccion", json={"filas": filas})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo"] == INFO_P3
    assert cuerpo["predicciones"] == [
        {"estacion_id": 1, "cluster": 1},
        {"estacion_id": 2, "cluster": 0},
    ]
    # El modelo recibe exactamente sus columnas, en orden y sin estacion_id.
    assert list(modelo.recibido.columns) == COLUMNAS_P3


def test_estacion_id_es_opcional(cliente):
    sin_id = {k: v for k, v in perfil().items() if k != "estacion_id"}
    respuesta = cliente.post("/p3/prediccion", json={"filas": [sin_id]})
    assert respuesta.status_code == 200
    assert respuesta.json()["predicciones"][0]["estacion_id"] is None


def test_clusters_acepta_perfiles_redondeados(cliente):
    # Un cliente que redondea a 4 decimales deja el bloque en 1 ± 0.0014 como máximo.
    fila = {k: round(v, 4) if isinstance(v, float) else v for k, v in perfil().items()}
    assert cliente.post("/p3/prediccion", json={"filas": [fila]}).status_code == 200


def test_info_del_modelo_de_p3_y_salud(cliente):
    assert cliente.get("/p3/modelo").json() == INFO_P3
    assert cliente.get("/modelo").json() == INFO_P1  # el endpoint de P1 no cambia
    assert cliente.get("/salud").json()["modelos"] == {"p1": "2", "p2": "1", "p3": "2"}


@pytest.mark.parametrize(
    "cambios",
    [
        {"sal_lun_manana": -0.1},
        {"lle_dom_noche": 1.5},
        {"sal_mar_tarde": "mucho"},
        {"columna_extra": 1},
        {"sal_lun_manana": 0.5},  # el bloque de salidas deja de sumar 1
        {"lle_lun_manana": 0.0},  # el bloque de llegadas deja de sumar 1
    ],
)
def test_rechaza_perfiles_invalidos(cliente, cambios):
    respuesta = cliente.post("/p3/prediccion", json={"filas": [perfil(**cambios)]})
    assert respuesta.status_code == 422


def test_rechaza_un_perfil_con_columnas_faltantes(cliente):
    incompleto = {k: v for k, v in perfil().items() if k != "sal_lun_manana"}
    assert cliente.post("/p3/prediccion", json={"filas": [incompleto]}).status_code == 422


def test_rechaza_solicitudes_vacias_o_con_columnas_de_p1(cliente):
    assert cliente.post("/p3/prediccion", json={"filas": []}).status_code == 422
    assert cliente.post("/p3/prediccion", json={"filas": [{"dia_semana": 2}]}).status_code == 422


@pytest.mark.parametrize("valor", ["NaN", "Infinity", "-Infinity"])
def test_rechaza_valores_no_finitos_con_422(cliente, valor):
    cuerpo = json.dumps({"filas": [perfil(sal_lun_manana=float(valor))]})
    assert valor in cuerpo
    respuesta = cliente.post(
        "/p3/prediccion", content=cuerpo, headers={"Content-Type": "application/json"}
    )
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "filas", 0, "sal_lun_manana"]


def test_p1_sigue_igual_con_p3_cargado(cliente):
    # /prediccion de P1 no usa el modelo de P3 ni se ve afectado por sus columnas.
    fila_p1 = {
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
    assert cliente.post("/prediccion", json={"filas": [fila_p1]}).status_code == 200
    assert cliente.post("/prediccion", json={"filas": [perfil()]}).status_code == 422
