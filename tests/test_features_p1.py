"""Pruebas de las variables de P1: conteos, historial sin información del futuro, feriados,
partición y la tabla estación-día que usa el entrenamiento."""

import numpy as np
import pandas as pd
import pytest

from mibici_mlops.config import FECHA_INICIO_PRUEBA, FECHA_INICIO_VALIDACION
from mibici_mlops.features import (
    COLUMNAS_P1,
    asignar_conjunto,
    conteos_diarios,
    es_feriado,
    historial_demanda,
    tabla_estacion_dia,
)

SEMILLA = 7
N_ESTACIONES = 4
N_DIAS = 40


def malla():
    """Conteos simulados de 4 estaciones durante 40 días, desde el 2024-03-01."""
    dias = pd.date_range("2024-03-01", periods=N_DIAS, freq="D")
    matriz = np.random.default_rng(SEMILLA).poisson(30, (N_ESTACIONES, len(dias)))
    return matriz.astype(np.float32), dias


def estaciones():
    return pd.DataFrame(
        {"dpcapacity": [10, 15, 20, 25], "latitude": 20.6, "longitude": -103.3},
        index=pd.Index([1, 2, 3, 4], name="id"),
    )


# conteos_diarios


def test_conteos_cuenta_viajes_por_estacion_y_dia():
    dias = pd.date_range("2024-03-01", periods=3, freq="D")
    viajes = pd.DataFrame(
        {
            "inicio": pd.to_datetime(
                ["2024-03-01 08:00", "2024-03-01 23:59", "2024-03-03 00:00", "2024-03-03 12:00"]
            ),
            "origen_id": [2, 2, 1, 2],
        }
    )
    matriz = conteos_diarios(viajes, pd.Index([1, 2]), dias)
    # El 2 de marzo no tiene viajes y queda en 0: la malla es completa.
    np.testing.assert_array_equal(matriz, [[0, 0, 1], [2, 0, 1]])
    assert matriz.dtype == np.float32


def test_conteos_descarta_estaciones_fuera_de_la_nomenclatura():
    dias = pd.date_range("2024-03-01", periods=2, freq="D")
    viajes = pd.DataFrame(
        {"inicio": pd.to_datetime(["2024-03-01", "2024-03-02"]), "origen_id": [1, 999]}
    )
    np.testing.assert_array_equal(conteos_diarios(viajes, pd.Index([1]), dias), [[1, 0]])


def test_conteos_sin_viajes_da_ceros():
    dias = pd.date_range("2024-03-01", periods=4, freq="D")
    viajes = pd.DataFrame(
        {"inicio": pd.Series([], dtype="datetime64[ns]"), "origen_id": pd.Series([], dtype=int)}
    )
    matriz = conteos_diarios(viajes, pd.Index([1, 2]), dias)
    assert matriz.shape == (2, 4)
    assert not matriz.any()


# historial_demanda


def test_historial_no_usa_informacion_del_dia_a_predecir():
    """Cambiar los viajes desde el día D no altera las variables del día D."""
    matriz, _ = malla()
    d = 20
    alterada = matriz.copy()
    alterada[:, d:] += 1000

    original, modificada = historial_demanda(matriz), historial_demanda(alterada)
    for nombre in original:
        np.testing.assert_array_equal(
            original[nombre][:, d],
            modificada[nombre][:, d],
            err_msg=f"{nombre} usa datos del día que se quiere predecir",
        )


def test_rezago_y_media_corresponden_al_corte():
    matriz, _ = malla()
    historial = historial_demanda(matriz)
    d, est = 20, 2
    assert historial["lag_7d"][est, d] == matriz[est, d - 7]
    np.testing.assert_allclose(historial["media_7d"][est, d], matriz[est, d - 7 : d].mean())


def test_primera_semana_sin_historial_es_nula():
    matriz, _ = malla()
    historial = historial_demanda(matriz)
    for nombre, valores in historial.items():
        assert valores.shape == matriz.shape
        assert np.isnan(valores[:, :7]).all(), nombre
        assert np.isfinite(valores[:, 7:]).all(), nombre


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="historial_demanda falla con mallas de menos de 7 días; en el proyecto la malla "
    "abarca años, así que no ocurre, pero el caso límite debería dar todo NaN.",
)
def test_historial_con_menos_de_una_semana_es_todo_nulo():
    historial = historial_demanda(np.ones((2, 5), dtype=np.float32))
    assert all(np.isnan(valores).all() for valores in historial.values())


# es_feriado


@pytest.mark.parametrize(
    "fecha, esperado",
    [
        ("2024-01-01", True),
        ("2024-02-05", True),  # primer lunes de febrero
        ("2027-02-01", True),  # el 1 de febrero cae en lunes: es el primer lunes
        ("2024-02-06", False),
        ("2024-03-18", True),  # tercer lunes de marzo
        ("2024-03-21", False),  # el 21 de marzo solo cuenta si cae en el tercer lunes
        ("2024-05-01", True),
        ("2024-09-16", True),
        ("2025-11-17", True),  # tercer lunes de noviembre
        ("2025-11-20", False),
        ("2024-12-25", True),
        ("2024-10-01", True),  # transmisión del Poder Ejecutivo (cada 6 años)
        ("2025-10-01", False),
        ("2024-07-15", False),
    ],
)
def test_feriados(fecha, esperado):
    assert es_feriado(pd.DatetimeIndex([fecha]))[0] == esperado


def test_feriado_ignora_la_hora():
    assert es_feriado(pd.DatetimeIndex(["2024-12-25 18:30"]))[0]


# asignar_conjunto


def test_asignar_conjunto_respeta_los_cortes_de_la_particion():
    """El día del corte ya pertenece al conjunto siguiente."""
    validacion = pd.Timestamp(FECHA_INICIO_VALIDACION)
    prueba = pd.Timestamp(FECHA_INICIO_PRUEBA)
    un_dia = pd.Timedelta(days=1)
    conjunto = asignar_conjunto(
        pd.Series([validacion - un_dia, validacion, prueba - un_dia, prueba])
    )
    assert list(conjunto) == ["entrenamiento", "validacion", "validacion", "prueba"]
    assert list(conjunto.categories) == ["entrenamiento", "validacion", "prueba"]


# tabla_estacion_dia


def test_tabla_tiene_las_columnas_del_modelo_sin_nulos_ni_duplicados():
    matriz, dias = malla()
    tabla = tabla_estacion_dia(matriz, dias, estaciones())
    assert {*COLUMNAS_P1, "estacion_id", "fecha", "viajes", "conjunto"} <= set(tabla.columns)
    assert tabla[[*COLUMNAS_P1, "viajes"]].notna().all().all()
    assert (tabla["viajes"] >= 0).all()
    assert not tabla.duplicated(["estacion_id", "fecha"]).any()


def test_tabla_sin_historial_suficiente_no_tiene_filas():
    """Los primeros 7 días no tienen una semana completa antes del corte."""
    matriz, dias = malla()
    tabla = tabla_estacion_dia(matriz, dias, estaciones())
    assert tabla["fecha"].min() == dias[7]
    assert len(tabla) == N_ESTACIONES * (N_DIAS - 7)


def test_tabla_objetivo_y_variables_de_la_estacion():
    matriz, dias = malla()
    tabla = tabla_estacion_dia(matriz, dias, estaciones())
    fila = tabla.iloc[0]
    est = estaciones().index.get_loc(fila["estacion_id"])
    assert fila["viajes"] == matriz[est, dias.get_loc(fila["fecha"])]
    assert fila["capacidad"] == estaciones()["dpcapacity"].iloc[est]


def test_tabla_variables_de_calendario():
    matriz, dias = malla()
    tabla = tabla_estacion_dia(matriz, dias, estaciones())
    assert (tabla["dia_semana"] == tabla["fecha"].dt.dayofweek).all()
    assert (tabla["mes"] == tabla["fecha"].dt.month).all()
    assert (tabla["es_fin_de_semana"] == (tabla["fecha"].dt.dayofweek >= 5)).all()
    # 2024-03-18 (tercer lunes de marzo) está dentro de la malla.
    assert tabla.loc[tabla["fecha"] == "2024-03-18", "es_feriado"].eq(1).all()
    assert tabla.loc[tabla["fecha"] == "2024-03-19", "es_feriado"].eq(0).all()


def test_estacion_sin_viajes_en_la_semana_previa_se_excluye():
    matriz, dias = malla()
    matriz[0, :] = 0
    tabla = tabla_estacion_dia(matriz, dias, estaciones())
    assert 1 not in set(tabla["estacion_id"])
