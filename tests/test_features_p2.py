"""Pruebas de las variables de P2: inventario simulado, ventanas, etiqueta, frecuencia de las
semanas anteriores (sin información del futuro), calendario y la tabla completa de `features p2`.

Usan viajes hechos a mano, así que cada resultado esperado se puede calcular en la cabeza. Las
reglas que se prueban están en references/decisiones_p2.md (decisiones 2, 4, 5 y 7).
"""

import numpy as np
import pandas as pd
import pytest

from mibici_mlops import features
from mibici_mlops.config import FECHA_INICIO, P2_INVENTARIO_INICIAL
from mibici_mlops.features import (
    CLASES_P2,
    COLUMNAS_P2,
    completar_ventanas,
    construir_movimientos,
    dia_de_simulacion,
    etiquetar,
    frecuencia_semanas_anteriores,
    p2,
    resumir_ventanas,
    simular_inventario,
    variables_calendario,
)

# Lunes 4 de marzo de 2024: dentro del periodo de análisis y lejos de feriados.
LUNES = pd.Timestamp("2024-03-04")


def momento(texto: str) -> pd.Timestamp:
    return pd.Timestamp(texto)


def viajes(*filas: tuple[int, int, str, str]) -> pd.DataFrame:
    """Viajes (origen, destino, inicio, fin) con las columnas que lee `features p2`."""
    return pd.DataFrame(
        [(o, d, momento(i), momento(f)) for o, d, i, f in filas],
        columns=["origen_id", "destino_id", "inicio", "fin"],
    )


def movimientos(*filas: tuple[int, str, int]) -> pd.DataFrame:
    """Movimientos (estación, momento, cambio) como los deja `construir_movimientos`."""
    return pd.DataFrame(
        [(e, momento(m), c) for e, m, c in filas], columns=["estacion", "momento", "cambio"]
    )


def ventanas_de(filas: list[tuple[int, str]], capacidad: dict[int, int]) -> pd.DataFrame:
    """Ventanas completas y etiquetadas de una lista de movimientos (estación, momento, cambio)."""
    capacidad = pd.Series(capacidad)
    simulados = simular_inventario(movimientos(*filas), capacidad)
    ventanas = completar_ventanas(resumir_ventanas(simulados), capacidad)
    ventanas["clase"] = etiquetar(ventanas)
    return ventanas.set_index("ventana")


# dia_de_simulacion


@pytest.mark.parametrize(
    "texto, dia",
    [
        ("2024-03-04 04:00", "2024-03-04"),  # el reinicio ya es del día nuevo
        ("2024-03-04 03:59:59", "2024-03-03"),  # un segundo antes es del día anterior
        ("2024-03-04 23:59", "2024-03-04"),
        ("2024-03-05 00:30", "2024-03-04"),  # la madrugada cuenta en el día anterior
    ],
)
def test_dia_de_simulacion_empieza_a_la_hora_de_reinicio(texto, dia):
    assert dia_de_simulacion(pd.Series([momento(texto)]))[0] == pd.Timestamp(dia)


# construir_movimientos


def test_cada_viaje_es_una_salida_en_el_origen_y_una_llegada_en_el_destino():
    resultado = construir_movimientos(
        viajes((1, 2, "2024-03-04 08:00", "2024-03-04 08:15")), np.array([1, 2])
    )
    assert sorted(resultado.itertuples(index=False)) == [
        (1, momento("2024-03-04 08:00"), -1),
        (2, momento("2024-03-04 08:15"), 1),
    ]


def test_llegada_desde_una_estacion_fuera_del_catalogo_si_cuenta():
    """El origen 99 no está en el catálogo: se descarta su salida, pero no la llegada a 1."""
    resultado = construir_movimientos(
        viajes((99, 1, "2024-03-04 08:00", "2024-03-04 08:15")), np.array([1])
    )
    assert list(resultado.itertuples(index=False)) == [(1, momento("2024-03-04 08:15"), 1)]


# simular_inventario


def test_inventario_acumula_dentro_del_dia_y_se_reinicia_a_las_4():
    capacidad = pd.Series({1: 10})
    resultado = simular_inventario(
        movimientos(
            (1, "2024-03-04 08:00", -1),
            (1, "2024-03-04 08:30", -1),
            (1, "2024-03-04 09:00", 1),
            (1, "2024-03-05 03:59", -1),  # todavía es el día 4
            (1, "2024-03-05 04:00", -1),  # día nuevo: vuelve a empezar en 5
        ),
        capacidad,
    )
    assert list(resultado["inventario"]) == [4, 3, 4, 3, 4]


def test_inventario_inicial_no_se_redondea_con_capacidad_impar():
    resultado = simular_inventario(movimientos((1, "2024-03-04 08:00", -1)), pd.Series({1: 7}))
    assert resultado["inventario"].iloc[0] == P2_INVENTARIO_INICIAL * 7 - 1 == 2.5


def test_movimientos_del_mismo_segundo_conservan_su_orden():
    """Con capacidad 2 la estación empieza con 1: si la salida va primero, llega a 0."""
    capacidad = pd.Series({1: 2})
    simulados = simular_inventario(
        construir_movimientos(
            viajes(
                (1, 2, "2024-03-04 08:00", "2024-03-04 08:10"),
                (2, 1, "2024-03-04 07:50", "2024-03-04 08:00"),  # llega en el mismo segundo
            ),
            np.array([1]),
        ),
        capacidad,
    )
    assert list(simulados["inventario"]) == [0, 1]


# resumir_ventanas


def test_resumen_por_ventana_de_2_horas():
    capacidad = pd.Series({1: 10})
    simulados = simular_inventario(
        movimientos(
            (1, "2024-03-04 08:00", -1),
            (1, "2024-03-04 08:30", -1),
            (1, "2024-03-04 09:59", 1),
            (1, "2024-03-04 10:00", 1),  # el límite ya es de la ventana siguiente
        ),
        capacidad,
    )
    resumen = resumir_ventanas(simulados).set_index("ventana")
    de_8_a_10 = resumen.loc[momento("2024-03-04 08:00")]
    assert de_8_a_10[["minimo", "maximo", "final", "neto"]].tolist() == [3, 4, 4, -1]
    assert resumen.loc[momento("2024-03-04 10:00"), "final"] == 5


# completar_ventanas y etiquetar


def test_estacion_vaciada_sigue_vacia_en_ventanas_sin_viajes():
    """Sin estas ventanas se perdería ~20% de las ventanas "se vacía" (decisión 5)."""
    filas = [(1, f"2024-03-04 08:{m:02d}", -1) for m in range(5)]  # 5 salidas: llega a 0
    filas.append((1, "2024-03-05 12:00", 1))  # otro día, para que la estación siga activa
    ventanas = ventanas_de(filas, {1: 10})

    dia_4 = ventanas[ventanas["dia"] == LUNES]
    assert len(dia_4) == 12  # de 04:00 a 02:00 del día siguiente
    desde_las_8 = dia_4.index >= momento("2024-03-04 08:00")
    assert dia_4.loc[~desde_las_8, "clase"].eq("sin_riesgo").all()
    # 08:00 llega a 0 y las 9 ventanas siguientes, sin viajes, siguen vacías.
    assert dia_4.loc[desde_las_8, "clase"].eq("se_vacia").all()
    # El día 5 se reinicia al 50%.
    assert ventanas.loc[momento("2024-03-05 04:00"), "final"] == 5
    assert ventanas.loc[momento("2024-03-05 04:00"), "clase"] == "sin_riesgo"


def test_ventanas_desde_el_primer_hasta_el_ultimo_dia_con_viajes():
    """Antes de su primer viaje la estación quizá no existía: no se cuenta como "sin riesgo"."""
    ventanas = ventanas_de(
        [(1, "2024-03-04 10:00", -1), (2, "2024-03-06 10:00", -1)], {1: 10, 2: 10}
    ).reset_index()
    por_estacion = ventanas.groupby("estacion")["ventana"].agg(["min", "max", "count"])
    primera, ultima, cuantas = por_estacion.loc[1]
    assert primera == momento("2024-03-04 04:00")
    assert ultima == momento("2024-03-05 02:00")
    assert cuantas == 12
    assert por_estacion.loc[2, "min"] == momento("2024-03-06 04:00")
    assert not ventanas.duplicated(["estacion", "ventana"]).any()


def test_se_descarta_el_dia_incompleto_anterior_a_fecha_inicio():
    """A las 02:00 del primer día todavía es el día anterior de la simulación."""
    inicio = pd.Timestamp(FECHA_INICIO)
    ventanas = ventanas_de(
        [
            (1, f"{inicio:%Y-%m-%d} 02:00", -1),
            (1, f"{inicio:%Y-%m-%d} 10:00", -1),
        ],
        {1: 10},
    )
    assert ventanas.index.min() == inicio + pd.Timedelta(hours=4)
    assert (ventanas["dia"] >= inicio).all()
    assert len(ventanas) == 12


def etiqueta(minimo, maximo, inicio, capacidad=10) -> str:
    fila = pd.DataFrame(
        {"minimo": [minimo], "maximo": [maximo], "inicio": [inicio], "capacidad": [capacidad]}
    )
    return etiquetar(fila).iloc[0]


@pytest.mark.parametrize(
    "minimo, maximo, inicio, esperada",
    [
        (3, 7, 5, "sin_riesgo"),
        (0, 5, 5, "se_vacia"),  # llegar justo a 0 ya es riesgo
        (-2, 5, 5, "se_vacia"),  # el inventario simulado puede quedar por debajo de 0
        (0.5, 5, 5, "sin_riesgo"),  # media bicicleta por encima: todavía no
        (5, 10, 5, "se_llena"),  # llegar justo a la capacidad ya es riesgo
        (5, 12, 5, "se_llena"),
        (5, 9.5, 5, "sin_riesgo"),
    ],
)
def test_umbrales_de_la_etiqueta(minimo, maximo, inicio, esperada):
    assert etiqueta(minimo, maximo, inicio) == esperada


@pytest.mark.parametrize(
    "inicio, esperada",
    [
        (2, "se_vacia"),  # empieza cerca de 0: lo primero que pasó fue vaciarse
        (5, "se_vacia"),  # justo a la mitad cuenta como abajo
        (8, "se_llena"),  # empieza cerca de la capacidad
    ],
)
def test_ventana_que_cumple_las_dos_reglas_gana_el_umbral_mas_cercano(inicio, esperada):
    assert etiqueta(minimo=0, maximo=10, inicio=inicio) == esperada


def test_etiqueta_solo_usa_las_tres_clases():
    rng = np.random.default_rng(43)
    ventanas = pd.DataFrame(
        {
            "minimo": rng.integers(-5, 10, 200),
            "maximo": rng.integers(0, 15, 200),
            "inicio": rng.integers(0, 10, 200),
            "capacidad": 10,
        }
    )
    assert set(etiquetar(ventanas)) <= set(CLASES_P2)


# frecuencia_semanas_anteriores


def semanas_de_lunes(clases: list[str], estacion: int = 1) -> pd.DataFrame:
    """Una ventana (lunes a las 08:00) por semana, con la clase de cada semana."""
    return pd.DataFrame(
        {
            "estacion": estacion,
            "ventana": [LUNES + pd.Timedelta(weeks=k, hours=8) for k in range(len(clases))],
            "clase": clases,
        }
    )


def test_frecuencia_es_la_fraccion_de_las_4_semanas_anteriores():
    clases = ["se_vacia", "se_vacia", "sin_riesgo", "se_llena", "sin_riesgo", "se_vacia"]
    frecuencias = frecuencia_semanas_anteriores(semanas_de_lunes(clases))
    # Semana 5 (índice 4): las 4 anteriores son vacia, vacia, sin_riesgo, llena.
    assert frecuencias.loc[4].tolist() == [0.5, 0.25, 0.25]
    # Semana 6: vacia, sin_riesgo, llena, sin_riesgo. La semana 1 ya no cuenta.
    assert frecuencias.loc[5].tolist() == [0.25, 0.25, 0.5]


def test_frecuencia_sin_historia_cuenta_como_sin_riesgo():
    """Las primeras semanas (o una estación nueva) no tienen 4 semanas anteriores."""
    frecuencias = frecuencia_semanas_anteriores(semanas_de_lunes(["se_vacia"] * 3))
    assert frecuencias["frec_se_vacia"].tolist() == [0, 0.25, 0.5]
    assert frecuencias["frec_sin_riesgo"].tolist() == [1, 0.75, 0.5]


def test_frecuencia_no_usa_la_ventana_que_se_predice():
    """Cambiar la clase de una ventana no cambia su propia frecuencia (no hay fuga)."""
    original = semanas_de_lunes(["sin_riesgo"] * 5)
    alterada = original.copy()
    alterada.loc[4, "clase"] = "se_vacia"
    pd.testing.assert_series_equal(
        frecuencia_semanas_anteriores(original).loc[4],
        frecuencia_semanas_anteriores(alterada).loc[4],
    )


def test_frecuencia_no_mezcla_estaciones_dias_ni_horas():
    estacion_1 = semanas_de_lunes(["se_vacia"] * 2, estacion=1)
    estacion_2 = semanas_de_lunes(["sin_riesgo"] * 2, estacion=2)
    martes = semanas_de_lunes(["sin_riesgo"] * 2).assign(
        ventana=lambda df: df["ventana"] + pd.Timedelta(days=1)
    )
    tabla = pd.concat([estacion_1, estacion_2, martes], ignore_index=True)
    frecuencias = frecuencia_semanas_anteriores(tabla)
    # Solo la segunda semana de la estación 1 tiene un lunes anterior "se vacía".
    assert frecuencias["frec_se_vacia"].tolist() == [0, 0.25, 0, 0, 0, 0]


def test_frecuencias_suman_1():
    rng = np.random.default_rng(43)
    tabla = semanas_de_lunes(list(rng.choice(CLASES_P2, 30)))
    assert np.allclose(frecuencia_semanas_anteriores(tabla).sum(axis=1), 1)


# variables_calendario


def test_variables_de_calendario():
    ventanas = pd.DataFrame(
        {
            "ventana": pd.to_datetime(
                ["2024-03-04 08:00", "2024-03-10 22:00", "2024-12-25 06:00", "2024-03-18 16:00"]
            )
        }
    )
    calendario = variables_calendario(ventanas)
    assert calendario["hora"].tolist() == [8, 22, 6, 16]
    assert calendario["dia_semana"].tolist() == [0, 6, 2, 0]  # 0 = lunes, 6 = domingo
    # Navidad y el tercer lunes de marzo son feriados.
    assert calendario["feriado"].tolist() == [False, False, True, True]


# p2: la tabla completa


@pytest.fixture
def archivos_p2(tmp_path):
    """Viajes de 2 estaciones durante 3 semanas y un catálogo con una estación sin capacidad."""
    rng = np.random.default_rng(43)
    filas = []
    for dia in pd.date_range(LUNES, periods=21, freq="D"):
        for _ in range(30):
            origen, destino = rng.choice([1, 2], size=2)
            inicio = dia + pd.Timedelta(
                hours=int(rng.integers(5, 23)), minutes=int(rng.integers(60))
            )
            filas.append((origen, destino, inicio, inicio + pd.Timedelta(minutes=10)))
    viajes_path = tmp_path / "viajes.parquet"
    pd.DataFrame(filas, columns=["origen_id", "destino_id", "inicio", "fin"]).to_parquet(
        viajes_path
    )
    estaciones_path = tmp_path / "estaciones.csv"
    pd.DataFrame({"id": [1, 2, 3], "dpcapacity": [6, 8, 0]}).to_csv(estaciones_path, index=False)
    return viajes_path, estaciones_path, tmp_path


def correr_p2(archivos, grupos: int) -> pd.DataFrame:
    viajes_path, estaciones_path, carpeta = archivos
    salida = carpeta / f"p2_{grupos}.parquet"
    p2(viajes_path=viajes_path, estaciones_path=estaciones_path, output_path=salida, grupos=grupos)
    return pd.read_parquet(salida)


def test_tabla_p2_columnas_sin_nulos_ni_duplicados(archivos_p2):
    tabla = correr_p2(archivos_p2, grupos=1)
    assert list(tabla.columns) == COLUMNAS_P2
    assert tabla.notna().all().all()
    assert not tabla.duplicated(["estacion", "ventana"]).any()
    assert set(tabla["clase"]) <= set(CLASES_P2)
    assert set(tabla["estacion"]) == {1, 2}  # la estación sin capacidad no entra
    # 12 ventanas por día y estación.
    assert (tabla.groupby(["estacion", "dia"]).size() == 12).all()


def test_tabla_p2_no_depende_de_los_grupos_de_estaciones(archivos_p2):
    """Se procesa por grupos para usar menos memoria; el resultado debe ser el mismo."""
    por_uno = correr_p2(archivos_p2, grupos=1)
    por_dos = correr_p2(archivos_p2, grupos=2)
    pd.testing.assert_frame_equal(
        por_uno.sort_values(["estacion", "ventana"], ignore_index=True),
        por_dos.sort_values(["estacion", "ventana"], ignore_index=True),
    )


def test_hora_de_reinicio_debe_coincidir_con_una_ventana(archivos_p2, monkeypatch):
    monkeypatch.setattr(features, "P2_HORA_REINICIO", 3)
    with pytest.raises(ValueError, match="P2_HORA_REINICIO"):
        correr_p2(archivos_p2, grupos=1)
