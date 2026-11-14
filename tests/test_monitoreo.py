"""Pruebas del monitoreo de P1 con Evidently: periodos, preparación de columnas y lectura del
drift por variable."""

from evidently import Report
import numpy as np
import pandas as pd

from mibici_mlops.features import COLUMNAS_P1
from mibici_mlops.monitoreo import (
    CATEGORICAS,
    UMBRAL_DRIFT,
    dataset,
    drift_por_mes,
    preparar,
    preset_drift,
    puntajes_drift,
    resumen,
    separar_periodos,
)

SEMILLA = 7


def tabla(desde: str, dias: int, estaciones: list[int], conjunto: str, nivel: float = 30.0):
    """Tabla estación-día simulada con las columnas de features p1."""
    rng = np.random.default_rng(SEMILLA)
    fechas = pd.date_range(desde, periods=dias, freq="D")
    filas = pd.MultiIndex.from_product([estaciones, fechas], names=["estacion_id", "fecha"])
    datos = filas.to_frame(index=False)
    n = len(datos)
    datos["viajes"] = rng.poisson(nivel, n).astype(np.int32)
    datos["dia_semana"] = datos["fecha"].dt.dayofweek.astype(np.int8)
    datos["mes"] = datos["fecha"].dt.month.astype(np.int8)
    datos["es_fin_de_semana"] = (datos["dia_semana"] >= 5).astype(np.int8)
    datos["es_feriado"] = np.int8(0)
    datos["capacidad"] = (10 + datos["estacion_id"]).astype(np.float32)
    datos["lat"] = np.float32(20.6) + datos["estacion_id"].astype(np.float32) / 100
    datos["lon"] = np.float32(-103.3) - datos["estacion_id"].astype(np.float32) / 100
    datos["lag_7d"] = rng.poisson(nivel, n).astype(np.float32)
    datos["media_7d"] = rng.normal(nivel, 3, n).astype(np.float32)
    datos["conjunto"] = pd.Categorical([conjunto] * n)
    return datos


def referencia_y_actual(nivel_actual: float = 30.0):
    referencia = tabla("2024-01-01", 120, [1, 2, 3, 4, 5], "entrenamiento")
    actual = tabla("2025-01-01", 120, [1, 2, 3, 4, 5, 6], "prueba", nivel_actual)
    return referencia, actual


def test_separar_periodos_usa_entrenamiento_y_las_fechas_del_periodo_actual():
    referencia, actual = referencia_y_actual()
    todo = pd.concat([referencia, actual], ignore_index=True)
    ref, act = separar_periodos(todo, "2025-02-01", "2025-02-28")
    assert set(ref["conjunto"].astype(str)) == {"entrenamiento"}
    assert len(ref) == len(referencia)
    assert act["fecha"].min() == pd.Timestamp("2025-02-01")
    assert act["fecha"].max() == pd.Timestamp("2025-02-28")


def test_preparar_convierte_categoricas_a_texto_y_conserva_las_variables():
    referencia, _ = referencia_y_actual()
    datos = preparar(referencia)
    assert set(COLUMNAS_P1) <= set(datos.columns)
    assert "conjunto" not in datos.columns
    for columna in CATEGORICAS:
        assert datos[columna].map(type).eq(str).all()


def test_sin_cambio_no_hay_drift_y_con_cambio_de_nivel_si():
    referencia, igual = referencia_y_actual()
    _, mayor = referencia_y_actual(nivel_actual=60.0)
    sin_cambio = puntajes_drift(Report([preset_drift()]).run(dataset(igual), dataset(referencia)))
    con_cambio = puntajes_drift(Report([preset_drift()]).run(dataset(mayor), dataset(referencia)))
    assert sin_cambio[["viajes", "lag_7d", "media_7d"]].lt(UMBRAL_DRIFT).all()
    assert con_cambio[["viajes", "lag_7d", "media_7d"]].ge(UMBRAL_DRIFT).all()


def test_drift_por_mes_compara_contra_el_mismo_mes_y_excluye_mes():
    referencia, actual = referencia_y_actual()
    por_mes = drift_por_mes(referencia, actual)
    assert list(por_mes.columns) == ["2025-01", "2025-02", "2025-03", "2025-04"]
    assert "mes" not in por_mes.index
    assert por_mes.notna().all().all()


def test_resumen_cuenta_estaciones_nuevas_y_variables_con_drift():
    referencia, actual = referencia_y_actual()
    puntajes = pd.Series({"viajes": 0.3, "lag_7d": 0.05, "mes": 0.0, "lat": 0.2})
    datos = resumen(referencia, actual, puntajes)
    assert datos["estaciones_nuevas"] == 1
    assert datos["filas_de_estaciones_nuevas"] == 120
    assert datos["variables_con_drift"] == ["viajes", "lat"]
    assert datos["drift_del_conjunto"] is True
