"""API REST que sirve dos modelos de MiBici: demanda por estación y día (P1) y grupo de una
estación según su perfil de uso (P3).

Los modelos se cargan una sola vez al arrancar (ver `cargar_modelo` en modelo.py): la copia del
`champion` de cada pregunta empaquetada en la imagen o, si se define MODELO_URI (P1) o
MODELO_URI_P3 (P3), la versión del registry. Basta con uno: los endpoints del modelo que falta
responden 503.

Endpoints:
    GET  /salud       el servicio está arriba y tiene los modelos cargados
    GET  /modelo      nombre, alias, versión y run de origen del modelo de P1
    POST /prediccion  P1: viajes esperados para una o varias estaciones-día
    GET  /clusters/modelo  nombre, alias, versión y run de origen del modelo de P3
    POST /clusters    P3: grupo de una o varias estaciones a partir de su perfil de uso

Uso local:
    uvicorn mibici_mlops.api.app:app --port 8000
La documentación interactiva queda en http://127.0.0.1:8000/docs.
"""

from contextlib import asynccontextmanager
import math
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
import numpy as np
import pandas as pd

from mibici_mlops.api.esquemas import (
    Cluster,
    InfoModelo,
    Prediccion,
    RespuestaClusters,
    RespuestaPrediccion,
    SolicitudClusters,
    SolicitudPrediccion,
)
from mibici_mlops.api.modelo import cargar_modelo
from mibici_mlops.features import COLUMNAS_P1, COLUMNAS_P3


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # Cada modelo es opcional: se sirve el que esté exportado (o con su MODELO_URI). Los endpoints
    # del que falta responden 503. Si no hay ninguno, la API no tiene nada que servir y no arranca.
    for pregunta, clave in (("p1", ""), ("p3", "_p3")):
        try:
            modelo, info = cargar_modelo(pregunta)
        except FileNotFoundError as error:
            logger.warning(f"Sin modelo de {pregunta.upper()}: {error}")
            modelo, info = None, None
        setattr(app.state, f"modelo{clave}", modelo)
        setattr(app.state, f"info{clave}", info)
    if app.state.modelo is None and app.state.modelo_p3 is None:
        raise FileNotFoundError(
            "No hay ningún modelo que servir: exporta al menos uno con `python -m "
            "mibici_mlops.api.modelo exportar --pregunta p1` (o p3), o define MODELO_URI "
            "(MODELO_URI_P3)"
        )
    yield


app = FastAPI(
    title="MiBici: demanda por estación (P1) y grupos de estaciones (P3)",
    description=(
        "P1: predice cuántos viajes iniciarán en cada estación de MiBici durante un día, a partir "
        "del calendario, la estación y sus viajes de la última semana. "
        "P3: asigna cada estación a un grupo según cómo se usa a lo largo de la semana. Equipo 43."
    ),
    version="1.1.0",
    lifespan=ciclo_de_vida,
)


def a_json_valido(valor):
    """El valor con NaN e Infinity cambiados por texto, que el JSON estándar no admite."""
    if isinstance(valor, float) and not math.isfinite(valor):
        return str(valor)
    if isinstance(valor, dict):
        return {k: a_json_valido(v) for k, v in valor.items()}
    if isinstance(valor, list | tuple):
        return [a_json_valido(v) for v in valor]
    return valor


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 con los errores de validación, como FastAPI, pero siempre serializable.

    Los errores repiten el valor recibido; si era NaN o Infinity, el manejador de FastAPI no
    puede convertirlo a JSON y la petición termina en 500 en lugar de 422.
    """
    return JSONResponse(
        status_code=422, content={"detail": a_json_valido(jsonable_encoder(exc.errors()))}
    )


def obtener_modelo(request: Request):
    """El modelo de P1 cargado al arrancar (None si no se exportó). Las pruebas lo sustituyen con
    `dependency_overrides`."""
    return request.app.state.modelo, request.app.state.info


def obtener_modelo_p3(request: Request):
    """El modelo de P3 cargado al arrancar (None si no se exportó). Igual que el de P1."""
    return request.app.state.modelo_p3, request.app.state.info_p3


Modelo = Annotated[tuple, Depends(obtener_modelo)]
ModeloP3 = Annotated[tuple, Depends(obtener_modelo_p3)]


def requerir(modelo: tuple, pregunta: str) -> tuple:
    """El modelo y su info, o 503 si la API arrancó sin el modelo de esa pregunta."""
    if modelo[0] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Esta API no tiene cargado el modelo de {pregunta}: expórtalo con `python -m "
            f"mibici_mlops.api.modelo exportar --pregunta {pregunta.lower()}` y reinicia el servicio",
        )
    return modelo


def con_tipos_del_modelo(tabla: pd.DataFrame, modelo) -> pd.DataFrame:
    """La tabla con los tipos de la firma del modelo, si la tiene."""
    esquema = modelo.metadata.get_input_schema() if modelo.metadata else None
    if esquema is not None and esquema.has_input_names():
        tabla = tabla.astype({c.name: c.type.to_pandas() for c in esquema.inputs})
    return tabla


def a_tabla(solicitud: SolicitudPrediccion, modelo) -> pd.DataFrame:
    """Las filas en el orden de COLUMNAS_P1 y con los tipos de la firma del modelo."""
    tabla = pd.DataFrame([fila.model_dump(include=set(COLUMNAS_P1)) for fila in solicitud.filas])
    return con_tipos_del_modelo(tabla[COLUMNAS_P1], modelo)


def a_tabla_p3(solicitud: SolicitudClusters, modelo) -> pd.DataFrame:
    """Los perfiles en el orden de COLUMNAS_P3 y con los tipos de la firma del modelo."""
    tabla = pd.DataFrame([fila.model_dump(include=set(COLUMNAS_P3)) for fila in solicitud.filas])
    return con_tipos_del_modelo(tabla[COLUMNAS_P3], modelo)


@app.get("/salud")
def salud(modelo: Modelo, modelo_p3: ModeloP3) -> dict:
    # La versión es None si la API arrancó sin ese modelo.
    return {
        "estado": "ok",
        "version_modelo": modelo[1]["version"] if modelo[1] else None,
        "version_modelo_p3": modelo_p3[1]["version"] if modelo_p3[1] else None,
    }


@app.get("/modelo")
def info_modelo(modelo: Modelo) -> InfoModelo:
    return InfoModelo(**requerir(modelo, "P1")[1])


@app.post("/prediccion")
def predecir(solicitud: SolicitudPrediccion, modelo: Modelo) -> RespuestaPrediccion:
    """Viajes esperados por estación-día. Los negativos se recortan a 0 (es un conteo)."""
    pyfunc, info = requerir(modelo, "P1")
    viajes = np.clip(np.asarray(pyfunc.predict(a_tabla(solicitud, pyfunc)), dtype=float), 0, None)
    return RespuestaPrediccion(
        modelo=InfoModelo(**info),
        predicciones=[
            Prediccion(estacion_id=fila.estacion_id, viajes=round(float(v), 2))
            for fila, v in zip(solicitud.filas, viajes)
        ],
    )


@app.get("/clusters/modelo")
def info_modelo_p3(modelo: ModeloP3) -> InfoModelo:
    return InfoModelo(**requerir(modelo, "P3")[1])


@app.post("/clusters")
def agrupar(solicitud: SolicitudClusters, modelo: ModeloP3) -> RespuestaClusters:
    """Grupo de cada estación según su perfil de uso (P3)."""
    pyfunc, info = requerir(modelo, "P3")
    grupos = np.asarray(pyfunc.predict(a_tabla_p3(solicitud, pyfunc))).astype(int)
    return RespuestaClusters(
        modelo=InfoModelo(**info),
        predicciones=[
            Cluster(estacion_id=fila.estacion_id, cluster=int(g))
            for fila, g in zip(solicitud.filas, grupos)
        ],
    )
