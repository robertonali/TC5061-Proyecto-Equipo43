"""Esquemas de entrada y salida de la API (Pydantic).

Una fila de entrada es una estación-día con las mismas variables que usa el modelo de P1
(COLUMNAS_P1 en features.py, tabla data/processed/p1_estacion_dia.parquet). Los rangos salen de
cómo se construyen esas variables: el día de la semana es `dt.dayofweek` (0 = lunes), las
banderas son 0 o 1, y la ubicación cae dentro del área de servicio de MiBici en Guadalajara.
Una fila que no cumple se rechaza con 422 antes de llegar al modelo.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Área de servicio de MiBici con margen: las estaciones del periodo de entrenamiento están entre
# 20.64 y 20.74 de latitud y entre -103.43 y -103.30 de longitud.
LAT_MIN, LAT_MAX = 20.5, 20.9
LON_MIN, LON_MAX = -103.6, -103.1
SABADO = 5


class EstacionDia(BaseModel):
    """Una estación en el día que se quiere predecir, con su historial de la última semana."""

    model_config = ConfigDict(
        extra="forbid",
        # El JSON que acepta FastAPI admite NaN e Infinity; ninguna variable puede tomarlos.
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "estacion_id": 512,
                    "dia_semana": 3,
                    "mes": 8,
                    "es_fin_de_semana": 0,
                    "es_feriado": 0,
                    "capacidad": 15,
                    "lat": 20.69256,
                    "lon": -103.36436,
                    "lag_7d": 4,
                    "media_7d": 4.0,
                }
            ]
        },
    )

    estacion_id: int | None = Field(
        None,
        description="Identificador de la estación. No lo usa el modelo; se devuelve tal cual.",
    )
    dia_semana: int = Field(ge=0, le=6, description="Día de la semana: 0 = lunes, 6 = domingo.")
    mes: int = Field(ge=1, le=12, description="Mes del día a predecir.")
    es_fin_de_semana: Literal[0, 1] = Field(description="1 si es sábado o domingo.")
    es_feriado: Literal[0, 1] = Field(description="1 si es día feriado.")
    capacidad: float = Field(gt=0, description="Anclajes de la estación.")
    lat: float = Field(ge=LAT_MIN, le=LAT_MAX, description="Latitud de la estación.")
    lon: float = Field(ge=LON_MIN, le=LON_MAX, description="Longitud de la estación.")
    lag_7d: float = Field(ge=0, description="Viajes que iniciaron en la estación hace 7 días.")
    media_7d: float = Field(
        ge=0, description="Promedio de viajes diarios de la estación en los 7 días anteriores."
    )

    @model_validator(mode="after")
    def fin_de_semana_coherente(self) -> "EstacionDia":
        if self.es_fin_de_semana != int(self.dia_semana >= SABADO):
            raise ValueError(
                "es_fin_de_semana debe ser 1 si y solo si dia_semana es 5 (sábado) o 6 (domingo)"
            )
        return self


class SolicitudPrediccion(BaseModel):
    """Las estaciones-día a predecir. El proceso diario manda todas las estaciones juntas."""

    model_config = ConfigDict(extra="forbid")

    filas: list[EstacionDia] = Field(min_length=1, max_length=5000)


class InfoModelo(BaseModel):
    """Qué versión del Model Registry está sirviendo la API."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    nombre: str
    alias: str | None = Field(description="None si se exportó por número de versión.")
    version: str
    run_id: str
    origen: Literal["imagen", "registry"]


class Prediccion(BaseModel):
    estacion_id: int | None
    viajes: float = Field(
        ge=0, description="Viajes esperados que iniciarán en la estación ese día."
    )


class RespuestaPrediccion(BaseModel):
    modelo: InfoModelo
    predicciones: list[Prediccion]
