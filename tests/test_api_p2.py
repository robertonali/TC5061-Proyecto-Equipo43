"""Pruebas del endpoint de P2: validación de entradas, formato de la respuesta y regla de la clase.

Usan un modelo falso con `predict_proba` en lugar del exportado del registry, así que corren sin
servidor de MLflow ni models/servicio/p2/. También cubren `cargar_modelo("p2")` y `exportar
--question p2` con un registry falso.
"""

import json
from types import SimpleNamespace
from typing import get_args

from fastapi.testclient import TestClient
import mlflow
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici_mlops.api import modelo as modulo_modelo
from mibici_mlops.api import p2
from mibici_mlops.api.app import app, obtener_modelo
from mibici_mlops.features import CLASES_P2, VARIABLES_P2

INFO_P1 = {
    "nombre": "mibici-equipo43-p1-demanda",
    "alias": "champion",
    "version": "3",
    "run_id": "abc123",
    "origen": "imagen",
}
INFO_P2 = {
    "nombre": "mibici-equipo43-p2-riesgo",
    "alias": "champion",
    "version": "1",
    "run_id": "def456",
    "origen": "imagen",
}
LOTE_DIARIO = 468 * 12  # estaciones activas × ventanas de 2 horas


class ModeloFalsoP2:
    """Devuelve como probabilidades las frecuencias de cada fila.

    p(se_vacia) = frec_se_vacia, p(se_llena) = frec_se_llena y el resto para sin_riesgo. Las clases
    van en el orden alfabético de scikit-learn, distinto al de CLASES_P2, para comprobar que la API
    las reordena. Recuerda la tabla que recibió.
    """

    classes_ = np.array(["se_llena", "se_vacia", "sin_riesgo"])

    def predict_proba(self, tabla: pd.DataFrame) -> np.ndarray:
        self.recibido = tabla
        se_vacia = tabla["frec_se_vacia"].to_numpy(dtype=float)
        se_llena = tabla["frec_se_llena"].to_numpy(dtype=float)
        return np.column_stack([se_llena, se_vacia, 1 - se_vacia - se_llena])


@pytest.fixture
def modelo():
    return ModeloFalsoP2()


@pytest.fixture
def cliente(modelo):
    # Sin `with`, TestClient no corre el arranque, que cargaría los modelos reales.
    app.dependency_overrides[obtener_modelo] = lambda: (None, INFO_P1)
    app.dependency_overrides[p2.obtener_modelo_p2] = lambda: (modelo, INFO_P2)
    yield TestClient(app)
    app.dependency_overrides.clear()


def fila(**cambios) -> dict:
    base = {
        "estacion_id": 2,
        "hora": 8,
        "dia_semana": 1,
        "feriado": False,
        "capacidad": 23,
        "frec_se_vacia": 0.5,
        "frec_se_llena": 0.25,
    }
    return {**base, **cambios}


def test_salud_reporta_la_version_de_los_dos_modelos(cliente):
    assert cliente.get("/salud").json() == {"estado": "ok", "modelos": {"p1": "3", "p2": "1"}}


def test_modelo_de_p2(cliente):
    assert cliente.get("/p2/modelo").json() == INFO_P2


def test_prediccion_devuelve_clase_y_probabilidades_por_fila(cliente, modelo):
    filas = [
        fila(estacion_id=1, frec_se_vacia=0.5, frec_se_llena=0.25),
        fila(estacion_id=7, hora=18, frec_se_vacia=0.0, frec_se_llena=0.75),
    ]
    respuesta = cliente.post("/p2/prediccion", json={"filas": filas})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelo"] == INFO_P2
    assert cuerpo["predicciones"] == [
        {
            "estacion_id": 1,
            "hora": 8,
            "clase": "se_vacia",
            "probabilidades": {"se_vacia": 0.5, "se_llena": 0.25, "sin_riesgo": 0.25},
        },
        {
            "estacion_id": 7,
            "hora": 18,
            "clase": "se_llena",
            "probabilidades": {"se_vacia": 0.0, "se_llena": 0.75, "sin_riesgo": 0.25},
        },
    ]
    # El modelo recibe exactamente sus columnas, en orden, con los tipos del entrenamiento y sin
    # estacion_id.
    assert list(modelo.recibido.columns) == VARIABLES_P2
    assert modelo.recibido.dtypes.astype(str).to_dict() == p2.TIPOS


def test_sin_riesgo_cuando_es_la_clase_mas_probable(cliente):
    respuesta = cliente.post(
        "/p2/prediccion", json={"filas": [fila(frec_se_vacia=0.25, frec_se_llena=0.0)]}
    )
    prediccion = respuesta.json()["predicciones"][0]
    assert prediccion["clase"] == "sin_riesgo"
    assert prediccion["probabilidades"] == {"se_vacia": 0.25, "se_llena": 0.0, "sin_riesgo": 0.75}


@pytest.mark.parametrize(
    ("frec_se_vacia", "frec_se_llena", "esperada"),
    [
        (0.5, 0.5, "se_vacia"),  # empate entre los dos riesgos: gana se_vacia
        (0.5, 0.0, "se_vacia"),  # empate se_vacia / sin_riesgo: gana el riesgo
        (0.0, 0.5, "se_llena"),  # empate se_llena / sin_riesgo: gana el riesgo
    ],
)
def test_en_empate_gana_el_riesgo(cliente, frec_se_vacia, frec_se_llena, esperada):
    filas = [fila(frec_se_vacia=frec_se_vacia, frec_se_llena=frec_se_llena)]
    respuesta = cliente.post("/p2/prediccion", json={"filas": filas})
    assert respuesta.json()["predicciones"][0]["clase"] == esperada


def test_probabilidades_redondeadas_a_tres_decimales(cliente):
    filas = [fila(frec_se_vacia=0.123456, frec_se_llena=0.654321)]
    probabilidades = cliente.post("/p2/prediccion", json={"filas": filas}).json()["predicciones"][
        0
    ]["probabilidades"]
    assert probabilidades == {"se_vacia": 0.123, "se_llena": 0.654, "sin_riesgo": 0.222}


def test_estacion_id_es_opcional_y_la_hora_se_devuelve(cliente):
    sin_id = {k: v for k, v in fila(hora=22).items() if k != "estacion_id"}
    respuesta = cliente.post("/p2/prediccion", json={"filas": [sin_id]})
    assert respuesta.status_code == 200
    prediccion = respuesta.json()["predicciones"][0]
    assert prediccion["estacion_id"] is None
    assert prediccion["hora"] == 22


def test_acepta_el_lote_diario_completo(cliente):
    filas = [fila(estacion_id=e, hora=h) for e in range(468) for h in range(0, 24, 2)]
    assert len(filas) == LOTE_DIARIO <= p2.MAX_FILAS
    respuesta = cliente.post("/p2/prediccion", json={"filas": filas})
    assert respuesta.status_code == 200
    assert len(respuesta.json()["predicciones"]) == LOTE_DIARIO


@pytest.mark.parametrize(
    "cambios",
    [
        {"hora": 7},  # las ventanas empiezan en horas pares
        {"hora": 24},
        {"hora": "8"},
        {"dia_semana": 7},
        {"dia_semana": -1},
        {"feriado": "tal vez"},
        {"capacidad": 0},
        {"capacidad": 12.5},  # los anclajes son enteros
        {"frec_se_vacia": 1.5},
        {"frec_se_llena": -0.25},
        {"frec_se_vacia": 0.75, "frec_se_llena": 0.5},  # suman más de 1
        {"columna_extra": 1},
    ],
)
def test_rechaza_filas_invalidas(cliente, cambios):
    respuesta = cliente.post("/p2/prediccion", json={"filas": [fila(**cambios)]})
    assert respuesta.status_code == 422


def test_acepta_frecuencias_que_suman_exactamente_uno(cliente):
    filas = [fila(frec_se_vacia=0.75, frec_se_llena=0.25)]
    assert cliente.post("/p2/prediccion", json={"filas": filas}).status_code == 200


@pytest.mark.parametrize("valor", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("campo", ["frec_se_vacia", "frec_se_llena", "capacidad"])
def test_rechaza_valores_no_finitos_con_422(cliente, campo, valor):
    # json.dumps escribe NaN e Infinity tal cual, como los mandaría un cliente descuidado.
    cuerpo = json.dumps({"filas": [fila(**{campo: float(valor)})]})
    assert valor in cuerpo
    respuesta = cliente.post(
        "/p2/prediccion", content=cuerpo, headers={"Content-Type": "application/json"}
    )
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "filas", 0, campo]


def test_rechaza_filas_incompletas_vacias_o_demasiadas(cliente):
    incompleta = {k: v for k, v in fila().items() if k != "frec_se_llena"}
    assert cliente.post("/p2/prediccion", json={"filas": [incompleta]}).status_code == 422
    assert cliente.post("/p2/prediccion", json={"filas": []}).status_code == 422
    demasiadas = [fila()] * (p2.MAX_FILAS + 1)
    assert cliente.post("/p2/prediccion", json={"filas": demasiadas}).status_code == 422


def test_el_esquema_coincide_con_el_modelo():
    assert set(get_args(p2.Clase)) == set(CLASES_P2)
    assert set(get_args(p2.Hora)) == set(range(0, 24, 2))
    assert list(p2.TIPOS) == VARIABLES_P2


def test_clase_predicha_sigue_la_regla_de_desempate_del_entrenamiento():
    probabilidades = pd.DataFrame(
        [[0.4, 0.4, 0.2], [0.2, 0.2, 0.6], [0.0, 0.5, 0.5]], columns=CLASES_P2
    )
    assert p2.clase_predicha(probabilidades) == ["se_vacia", "sin_riesgo", "se_llena"]


# cargar_modelo y exportar para P2


class ClienteFalso:
    """MlflowClient con una sola versión registrada: la 3, con alias champion."""

    version = SimpleNamespace(version="3", run_id="abc123")

    def get_model_version_by_alias(self, nombre, alias):
        assert alias == "champion"
        return self.version

    def get_model_version(self, nombre, version):
        assert version == "3"
        return self.version


def test_servicio_desconocido():
    with pytest.raises(ValueError, match="opciones: p1, p2"):
        modulo_modelo.servicio_de("p9")


def test_cargar_modelo_p2_desde_el_registry_en_vivo(monkeypatch):
    uri = "models:/mibici-equipo43-p2-riesgo@champion"
    monkeypatch.setenv("MODELO_URI_P2", uri)
    monkeypatch.setattr(modulo_modelo, "MlflowClient", ClienteFalso)
    monkeypatch.setitem(modulo_modelo.SERVICIOS["p2"], "cargador", lambda u: ("cargado", u))

    modelo, info = modulo_modelo.cargar_modelo("p2")

    assert modelo == ("cargado", uri)
    assert info == {
        "nombre": "mibici-equipo43-p2-riesgo",
        "alias": "champion",
        "version": "3",
        "run_id": "abc123",
        "origen": "registry",
    }


def test_cargar_modelo_p2_desde_la_copia_exportada(monkeypatch, tmp_path):
    monkeypatch.delenv("MODELO_URI_P2", raising=False)
    carpeta = tmp_path / "p2"
    carpeta.mkdir()
    (carpeta / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    (carpeta / "version.json").write_text(
        json.dumps({k: INFO_P2[k] for k in ("nombre", "alias", "version", "run_id")}),
        encoding="utf-8",
    )
    monkeypatch.setitem(modulo_modelo.SERVICIOS["p2"], "directorio", carpeta)
    monkeypatch.setitem(modulo_modelo.SERVICIOS["p2"], "cargador", lambda ruta: ("cargado", ruta))

    modelo, info = modulo_modelo.cargar_modelo("p2")

    assert modelo == ("cargado", str(carpeta))
    assert info == INFO_P2


def test_cargar_modelo_p2_sin_exportar_dice_que_hacer(monkeypatch, tmp_path):
    monkeypatch.delenv("MODELO_URI_P2", raising=False)
    monkeypatch.setitem(modulo_modelo.SERVICIOS["p2"], "directorio", tmp_path / "p2")
    with pytest.raises(FileNotFoundError, match="--question p2"):
        modulo_modelo.cargar_modelo("p2")


def test_exportar_por_pregunta_usa_el_champion_y_la_carpeta_de_p2(monkeypatch, tmp_path):
    llamadas = {}

    def metadatos_falsos(uri):
        llamadas["uri"] = uri
        return {k: INFO_P2[k] for k in ("nombre", "alias", "version", "run_id")}

    def descarga_falsa(artifact_uri, dst_path):
        llamadas["descarga"] = artifact_uri
        llamadas["destino"] = dst_path

    monkeypatch.setattr(modulo_modelo, "metadatos", metadatos_falsos)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", descarga_falsa)
    monkeypatch.setitem(modulo_modelo.SERVICIOS["p2"], "directorio", tmp_path / "p2")

    resultado = CliRunner().invoke(modulo_modelo.app, ["exportar", "--question", "p2"])

    assert resultado.exit_code == 0, resultado.output
    assert llamadas["uri"] == "models:/mibici-equipo43-p2-riesgo@champion"
    # Se descarga la versión fija a la que apunta el alias, no el alias.
    assert llamadas["descarga"] == "models:/mibici-equipo43-p2-riesgo/1"
    assert llamadas["destino"] == str(tmp_path / "p2")
    version = json.loads((tmp_path / "p2" / "version.json").read_text(encoding="utf-8"))
    assert version["alias"] == "champion" and version["version"] == "1"


def test_exportar_rechaza_una_pregunta_sin_servicio():
    resultado = CliRunner().invoke(modulo_modelo.app, ["exportar", "--question", "p9"])
    assert resultado.exit_code != 0
    assert "opciones: p1, p2" in resultado.output
