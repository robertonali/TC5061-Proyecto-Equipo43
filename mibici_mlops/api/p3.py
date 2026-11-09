"""P3 en la API: grupo de una estación según su perfil de uso a lo largo de la semana.

Sirve el `champion` de `mibici-equipo43-p3-clusters` (P3_SERVICIO_MODELO_URI en config.py). Se
carga con `pyfunc`: el modelo solo asigna el grupo, no hay probabilidades que devolver.

Endpoints (app.py monta este router con el prefijo /p3):
    GET  /p3/modelo      nombre, alias, versión y run de origen del modelo de P3
    POST /p3/prediccion  grupo de una o varias estaciones a partir de su perfil de uso

Cada fila es el perfil de uso de una estación: 28 fracciones de sus salidas (`sal_*`) y 28 de sus
llegadas (`lle_*`), una por día de la semana y franja del día (COLUMNAS_P3 en features.py, tabla
data/processed/p3_perfiles.parquet), y opcionalmente `estacion_id`. Cada bloque de 28 suma 1,
porque son proporciones del total de viajes de la estación. Una fila que no cumple se rechaza con
422 antes de llegar al modelo.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from mibici_mlops.api.esquemas import InfoModelo
from mibici_mlops.api.modelo import cargar_modelo
from mibici_mlops.features import COLUMNAS_P3, columnas_perfil

# Las fracciones del perfil se calculan con float64 y suman 1 sin más error que el de redondeo;
# la tolerancia deja pasar perfiles que un cliente redondeó a 4 o 5 decimales.
TOLERANCIA_SUMA = 1e-3
# Hay del orden de 300 estaciones con suficientes viajes para tener perfil (P3_MIN_VIAJES); el
# margen deja mandar todas en una petición aunque crezca la red.
MAX_FILAS = 5000


def ejemplo_perfil() -> dict:
    """Un perfil válido para la documentación interactiva: viajes repartidos por igual."""
    fraccion = 1 / len(columnas_perfil("sal"))
    return {"estacion_id": 512, **dict.fromkeys(COLUMNAS_P3, fraccion)}


class _PerfilBase(BaseModel):
    """Base del perfil de uso de una estación. Los 56 campos se agregan en `PerfilEstacion`."""

    model_config = ConfigDict(
        extra="forbid",
        # El JSON que acepta FastAPI admite NaN e Infinity; ninguna fracción puede tomarlos.
        allow_inf_nan=False,
        json_schema_extra={"examples": [ejemplo_perfil()]},
    )

    estacion_id: int | None = Field(
        None,
        description="Identificador de la estación. No lo usa el modelo; se devuelve tal cual.",
    )

    @model_validator(mode="after")
    def cada_bloque_suma_uno(self) -> "_PerfilBase":
        for prefijo, nombre in (("sal", "salidas"), ("lle", "llegadas")):
            suma = sum(getattr(self, columna) for columna in columnas_perfil(prefijo))
            if abs(suma - 1) > TOLERANCIA_SUMA:
                raise ValueError(
                    f"Las 28 fracciones de {nombre} ({prefijo}_*) deben sumar 1, no {suma:.4f}"
                )
        return self


PerfilEstacion = create_model(
    "PerfilEstacion",
    __base__=_PerfilBase,
    __doc__=(
        "Perfil de uso de una estación: la fracción de sus salidas (`sal_<día>_<franja>`) y de sus "
        "llegadas (`lle_<día>_<franja>`) en cada día de la semana y franja del día "
        "(madrugada 0-6 h, mañana 6-12 h, tarde 12-18 h, noche 18-24 h)."
    ),
    **{
        columna: (
            float,
            Field(
                ge=0,
                le=1,
                description=(
                    f"Fracción de {'las salidas' if columna.startswith('sal') else 'las llegadas'}"
                    f" de la estación en {columna.split('_', 1)[1].replace('_', ', ')}."
                ),
            ),
        )
        for columna in COLUMNAS_P3
    },
)


class SolicitudGrupos(BaseModel):
    """Los perfiles de las estaciones a agrupar. Pueden mandarse todas juntas."""

    model_config = ConfigDict(extra="forbid")

    filas: list[PerfilEstacion] = Field(min_length=1, max_length=MAX_FILAS)


class PrediccionGrupo(BaseModel):
    estacion_id: int | None
    cluster: int = Field(
        ge=0,
        description=(
            "Número del grupo al que pertenece la estación. Los números son del modelo servido: "
            "no se conservan entre versiones (mira `version` en `modelo`)."
        ),
    )


class RespuestaGrupos(BaseModel):
    modelo: InfoModelo
    predicciones: list[PrediccionGrupo]


def cargar() -> tuple[object, dict]:
    """El modelo de P3 (pyfunc) y sus metadatos."""
    return cargar_modelo("p3")


router = APIRouter(prefix="/p3", tags=["P3: grupos de estaciones"])


def obtener_modelo_p3(request: Request):
    """El modelo de P3 cargado al arrancar. Las pruebas lo sustituyen con `dependency_overrides`."""
    return request.app.state.modelo_p3, request.app.state.info_p3


ModeloP3 = Annotated[tuple, Depends(obtener_modelo_p3)]


def a_tabla(solicitud: SolicitudGrupos) -> pd.DataFrame:
    """Los perfiles con las columnas del modelo, en su orden y en float64 como en train_p3.py."""
    tabla = pd.DataFrame([fila.model_dump(include=set(COLUMNAS_P3)) for fila in solicitud.filas])
    return tabla[COLUMNAS_P3].astype("float64")


@router.get("/modelo")
def info_modelo(modelo: ModeloP3) -> InfoModelo:
    return InfoModelo(**modelo[1])


@router.post("/prediccion")
def predecir(solicitud: SolicitudGrupos, modelo: ModeloP3) -> RespuestaGrupos:
    """Grupo de cada estación según su perfil de uso."""
    pyfunc, info = modelo
    grupos = np.asarray(pyfunc.predict(a_tabla(solicitud))).astype(int)
    return RespuestaGrupos(
        modelo=InfoModelo(**info),
        predicciones=[
            PrediccionGrupo(estacion_id=fila.estacion_id, cluster=int(g))
            for fila, g in zip(solicitud.filas, grupos)
        ],
    )
