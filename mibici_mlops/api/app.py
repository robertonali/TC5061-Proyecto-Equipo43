"""API REST que sirve los modelos en producción del equipo 43.

- P1: viajes que iniciarán en cada estación el día siguiente (este archivo).
- P2: riesgo de que una estación se vacíe o se llene en una ventana de 2 horas (p2.py, con el
  prefijo /p2).

Cada modelo se carga una sola vez al arrancar (ver `cargar_modelo` en modelo.py): la copia del
`champion` empaquetada en la imagen o, si se define su variable de entorno (MODELO_URI,
MODELO_URI_P2), la versión del registry. Si falta alguno, la API no arranca.

Endpoints:
    GET  /salud          el servicio está arriba y qué versión tiene cargada de cada modelo
    GET  /modelo         nombre, alias, versión y run de origen del modelo de P1
    POST /prediccion     viajes esperados para una o varias estaciones-día
    GET  /p2/modelo      lo mismo para el modelo de P2
    POST /p2/prediccion  clase de riesgo y probabilidades para una o varias estaciones-ventana

Uso local:
    uvicorn mibici_mlops.api.app:app --port 8000
La documentación interactiva queda en http://127.0.0.1:8000/docs.
"""

from contextlib import asynccontextmanager
import math
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

from mibici_mlops.api import p2
from mibici_mlops.api.esquemas import (
    InfoModelo,
    Prediccion,
    RespuestaPrediccion,
    SolicitudPrediccion,
)
from mibici_mlops.api.modelo import cargar_modelo
from mibici_mlops.features import COLUMNAS_P1


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    app.state.modelo, app.state.info = cargar_modelo("p1")
    app.state.modelo_p2, app.state.info_p2 = p2.cargar()
    yield


app = FastAPI(
    title="MiBici: modelos del equipo 43",
    description=(
        "Sirve los modelos en producción (`champion`) del Model Registry. P1: cuántos viajes "
        "iniciarán en cada estación de MiBici durante un día, a partir del calendario, la estación "
        "y sus viajes de la última semana. P2: si una estación se quedará sin bicicletas o sin "
        "anclajes en una ventana de 2 horas, con la probabilidad de cada clase."
    ),
    version="1.1.0",
    lifespan=ciclo_de_vida,
)
# Cada pregunta agrega su router en su propio módulo, con su prefijo (/p2, /p3…).
app.include_router(p2.router)

TAG_P1 = "P1: demanda por estación y día"


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
    """El modelo cargado al arrancar. Las pruebas lo sustituyen con `dependency_overrides`."""
    return request.app.state.modelo, request.app.state.info


Modelo = Annotated[tuple, Depends(obtener_modelo)]


def a_tabla(solicitud: SolicitudPrediccion, modelo) -> pd.DataFrame:
    """Las filas en el orden de COLUMNAS_P1 y con los tipos de la firma del modelo."""
    tabla = pd.DataFrame([fila.model_dump(include=set(COLUMNAS_P1)) for fila in solicitud.filas])
    tabla = tabla[COLUMNAS_P1]
    esquema = modelo.metadata.get_input_schema() if modelo.metadata else None
    if esquema is not None and esquema.has_input_names():
        tabla = tabla.astype({c.name: c.type.to_pandas() for c in esquema.inputs})
    return tabla


@app.get("/salud", tags=["Servicio"])
def salud(modelo: Modelo, modelo_p2: p2.ModeloP2) -> dict:
    """El servicio está arriba y qué versión del registry tiene cargada de cada modelo."""
    return {"estado": "ok", "modelos": {"p1": modelo[1]["version"], "p2": modelo_p2[1]["version"]}}


@app.get("/modelo", tags=[TAG_P1])
def info_modelo(modelo: Modelo) -> InfoModelo:
    return InfoModelo(**modelo[1])


@app.post("/prediccion", tags=[TAG_P1])
def predecir(solicitud: SolicitudPrediccion, modelo: Modelo) -> RespuestaPrediccion:
    """Viajes esperados por estación-día. Los negativos se recortan a 0 (es un conteo)."""
    pyfunc, info = modelo
    viajes = np.clip(np.asarray(pyfunc.predict(a_tabla(solicitud, pyfunc)), dtype=float), 0, None)
    return RespuestaPrediccion(
        modelo=InfoModelo(**info),
        predicciones=[
            Prediccion(estacion_id=fila.estacion_id, viajes=round(float(v), 2))
            for fila, v in zip(solicitud.filas, viajes)
        ],
    )
