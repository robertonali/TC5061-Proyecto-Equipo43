"""P2 en la API: riesgo de que una estación se vacíe o se llene en una ventana de 2 horas.

Sirve el `champion` de `mibici-equipo43-p2-riesgo` (P2_SERVICIO_MODELO_URI en config.py), el
HistGradientBoosting con pesos de clase de la decisión 9 de P2. Se carga con `mlflow.sklearn` y
no con `pyfunc` para tener `predict_proba`: además de la clase, la respuesta trae la probabilidad
de cada una, para que el equipo de rebalanceo ordene las estaciones por riesgo (decisión 11).

Endpoints (app.py monta este router con el prefijo /p2):
    GET  /p2/modelo      nombre, alias, versión y run de origen del modelo de P2
    POST /p2/prediccion  clase y probabilidades para una o varias estaciones-ventana

Cada fila de entrada trae las 6 variables del modelo (VARIABLES_P2 en features.py), que se conocen
a las 4:00 del día que se predice (decisión 8), y opcionalmente `estacion_id`. Los rangos salen de
cómo construye esas variables features.py: las ventanas de 2 horas empiezan en horas pares, el día
de la semana es `dt.dayofweek` (0 = lunes) y las frecuencias son fracciones de las 4 semanas
anteriores, así que entre las dos no pueden pasar de 1. Una fila que no cumple se rechaza con 422
antes de llegar al modelo.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mibici_mlops.api.esquemas import InfoModelo
from mibici_mlops.api.modelo import cargar_modelo
from mibici_mlops.features import CLASES_P2, VARIABLES_P2

# Las mismas clases de CLASES_P2, escritas para que aparezcan en el esquema de /docs (hay una
# prueba que comprueba que coinciden).
Clase = Literal["se_vacia", "se_llena", "sin_riesgo"]
# Horas en que empieza una ventana de 2 horas (P2_VENTANA en config.py).
Hora = Literal[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
# Tipos con que se entrenó el modelo (los de la tabla de features.py y de la firma del modelo).
TIPOS = {
    "hora": "int32",
    "dia_semana": "int32",
    "feriado": "bool",
    "capacidad": "int64",
    "frec_se_vacia": "float64",
    "frec_se_llena": "float64",
}
# El lote de las 4:00 son las 12 ventanas del día de cada una de las 468 estaciones activas
# (5,616 filas); el margen deja crecer el número de estaciones.
MAX_FILAS = 10_000
TOLERANCIA = 1e-9
DECIMALES = 3


class EstacionVentana(BaseModel):
    """Una estación en la ventana de 2 horas que se quiere predecir, con su historial."""

    model_config = ConfigDict(
        extra="forbid",
        # El JSON que acepta FastAPI admite NaN e Infinity; ninguna variable puede tomarlos.
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "estacion_id": 2,
                    "hora": 8,
                    "dia_semana": 1,
                    "feriado": False,
                    "capacidad": 23,
                    "frec_se_vacia": 0.5,
                    "frec_se_llena": 0.0,
                }
            ]
        },
    )

    estacion_id: int | None = Field(
        None,
        description="Identificador de la estación. No lo usa el modelo; se devuelve tal cual.",
    )
    hora: Hora = Field(description="Hora en que empieza la ventana de 2 horas: 0, 2, 4, …, 22.")
    dia_semana: int = Field(ge=0, le=6, description="Día de la semana: 0 = lunes, 6 = domingo.")
    feriado: bool = Field(description="true si la ventana cae en un día feriado.")
    capacidad: int = Field(gt=0, description="Anclajes de la estación.")
    frec_se_vacia: float = Field(
        ge=0,
        le=1,
        description=(
            "Fracción de las 4 semanas anteriores en que la misma estación, el mismo día de la "
            "semana y la misma ventana se quedó sin bicicletas."
        ),
    )
    frec_se_llena: float = Field(
        ge=0,
        le=1,
        description=(
            "Fracción de las 4 semanas anteriores en que la misma estación, el mismo día de la "
            "semana y la misma ventana se quedó sin anclajes."
        ),
    )

    @model_validator(mode="after")
    def frecuencias_coherentes(self) -> "EstacionVentana":
        if self.frec_se_vacia + self.frec_se_llena > 1 + TOLERANCIA:
            raise ValueError(
                "frec_se_vacia + frec_se_llena no puede pasar de 1: son fracciones de las mismas "
                "semanas y una ventana no se vacía y se llena a la vez"
            )
        return self


class SolicitudRiesgo(BaseModel):
    """Las estaciones-ventana a predecir. El lote de las 4:00 manda las 12 ventanas del día."""

    model_config = ConfigDict(extra="forbid")

    filas: list[EstacionVentana] = Field(min_length=1, max_length=MAX_FILAS)


class Probabilidades(BaseModel):
    """Probabilidad de cada clase; suman 1 salvo por el redondeo."""

    se_vacia: float = Field(ge=0, le=1)
    se_llena: float = Field(ge=0, le=1)
    sin_riesgo: float = Field(ge=0, le=1)


class PrediccionRiesgo(BaseModel):
    estacion_id: int | None
    hora: Hora = Field(description="La ventana de la fila de entrada, para identificarla.")
    clase: Clase = Field(
        description=(
            "Clase más probable. En empate gana el riesgo: se_vacia, luego se_llena (decisión 7)."
        )
    )
    probabilidades: Probabilidades


class RespuestaRiesgo(BaseModel):
    modelo: InfoModelo
    predicciones: list[PrediccionRiesgo]


def cargar() -> tuple[object, dict]:
    """El modelo de P2 como lo necesita la API (con `predict_proba`) y sus metadatos."""
    return cargar_modelo("p2")


router = APIRouter(prefix="/p2", tags=["P2: riesgo por estación y ventana"])


def obtener_modelo_p2(request: Request):
    """El modelo de P2 cargado al arrancar. Las pruebas lo sustituyen con `dependency_overrides`."""
    return request.app.state.modelo_p2, request.app.state.info_p2


ModeloP2 = Annotated[tuple, Depends(obtener_modelo_p2)]


def a_tabla(solicitud: SolicitudRiesgo) -> pd.DataFrame:
    """Las filas con las columnas del modelo, en su orden y con los tipos del entrenamiento."""
    tabla = pd.DataFrame([fila.model_dump(include=set(VARIABLES_P2)) for fila in solicitud.filas])
    return tabla[VARIABLES_P2].astype(TIPOS)


def probabilidades_por_clase(modelo, tabla: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada clase en el orden de CLASES_P2 (como en train_p2.py)."""
    return pd.DataFrame(modelo.predict_proba(tabla), columns=list(modelo.classes_))[CLASES_P2]


def clase_predicha(probabilidades: pd.DataFrame) -> list[str]:
    """La clase con mayor probabilidad; en empate, la primera de CLASES_P2 (decisión 7).

    Es la misma regla que `predecir` en train_p2.py, con la que se evaluó el modelo: ante la duda,
    avisar del riesgo cuesta menos que callarlo.
    """
    indices = probabilidades[CLASES_P2].to_numpy().argmax(axis=1)
    return [CLASES_P2[i] for i in indices]


@router.get("/modelo")
def info_modelo(modelo: ModeloP2) -> InfoModelo:
    return InfoModelo(**modelo[1])


@router.post("/prediccion")
def predecir(solicitud: SolicitudRiesgo, modelo: ModeloP2) -> RespuestaRiesgo:
    """Clase de riesgo y probabilidad de cada clase para cada estación-ventana."""
    estimador, info = modelo
    probabilidades = probabilidades_por_clase(estimador, a_tabla(solicitud))
    clases = clase_predicha(probabilidades)
    return RespuestaRiesgo(
        modelo=InfoModelo(**info),
        predicciones=[
            PrediccionRiesgo(
                estacion_id=fila.estacion_id,
                hora=fila.hora,
                clase=clase,
                probabilidades=Probabilidades(
                    **{c: round(float(p), DECIMALES) for c, p in zip(CLASES_P2, valores)}
                ),
            )
            for fila, clase, valores in zip(solicitud.filas, clases, probabilidades.to_numpy())
        ],
    )
