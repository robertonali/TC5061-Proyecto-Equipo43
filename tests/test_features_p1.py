"""Pruebas de las variables de P1, sobre todo que no usen información del futuro."""

import numpy as np
import pandas as pd

from mibici_mlops.features import es_feriado, historial_demanda, tabla_estacion_dia

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


def test_historial_no_usa_informacion_del_dia_a_predecir():
    """Cambiar los viajes desde el día D no altera las variables del día D."""
    matriz, _ = malla()
    d = 20

    original = historial_demanda(matriz)
    alterada = matriz.copy()
    alterada[:, d:] += 1000
    modificada = historial_demanda(alterada)

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
    esperado = matriz[est, d - 7 : d].mean()  # días d-7 a d-1
    np.testing.assert_allclose(historial["media_7d"][est, d], esperado, rtol=1e-5)


def test_sin_historial_suficiente_no_hay_filas():
    """Los primeros 7 días de la malla no tienen una semana completa antes del corte."""
    matriz, dias = malla()
    tabla = tabla_estacion_dia(matriz, dias, estaciones())
    assert tabla["fecha"].min() >= dias[0] + pd.Timedelta(days=7)
    assert tabla[["lag_7d", "media_7d"]].notna().all().all()
    # La variable objetivo es el conteo de ese mismo día.
    fila = tabla.iloc[0]
    est = int(np.flatnonzero(estaciones().index == fila["estacion_id"])[0])
    assert fila["viajes"] == matriz[est, dias.get_loc(fila["fecha"])]


def test_estacion_sin_viajes_en_la_semana_previa_se_excluye():
    matriz, dias = malla()
    matriz[0, :] = 0
    tabla = tabla_estacion_dia(matriz, dias, estaciones())
    assert 1 not in set(tabla["estacion_id"])


def test_feriados():
    fechas = pd.DatetimeIndex(
        ["2024-01-01", "2024-02-05", "2024-03-18", "2024-09-16", "2024-10-01", "2025-10-01"]
    )
    esperado = [True, True, True, True, True, False]
    assert es_feriado(fechas).tolist() == esperado
