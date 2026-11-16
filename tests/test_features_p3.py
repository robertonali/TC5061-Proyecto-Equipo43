"""Pruebas de features de P3: franjas, perfil de uso y separación entre periodos."""

import pandas as pd
import numpy as np
import pytest

from mibici_mlops.features import (
    columnas_perfil,
    franja_del_dia,
    p3,
    p3_ampliado,
    perfil_de_uso,
    perfiles_estaciones,
)
from mibici_mlops.config import FECHA_INICIO, FECHA_INICIO_PRUEBA, P3_MIN_VIAJES

PERIODOS = {
    "entrenamiento": "2023-03-06",
    "validacion": "2025-10-06",
    "prueba": "2026-04-06",
}

def crear_viajes_de_prueba():
    """Ida y vuelta entre las estaciones 1 y 2 en cada periodo, con suficientes viajes."""
    viajes = []
    for periodo in PERIODOS.values():
        inicio = pd.date_range(periodo, periods=P3_MIN_VIAJES, freq="37min")
        viajes.append({
                        "inicio": inicio,
                        "fin": inicio + pd.Timedelta(minutes=15),
                        "origen_id": 1,
                        "destino_id": 2,
                    })
        viajes.append({
                        "inicio": inicio,
                        "fin": inicio + pd.Timedelta(minutes=15),
                        "origen_id": 2,
                        "destino_id": 1,
                    })
    # Cada dict tiene columnas de 500 valores: se convierte en su propio DataFrame y se unen:
    return pd.concat([pd.DataFrame(v) for v in viajes], ignore_index=True)

def test_p3_genera_la_tabla_de_perfiles(tmp_path):
    viajes_path = tmp_path / "viajes.parquet"
    estaciones_path = tmp_path / "estaciones.csv"
    output_path = tmp_path / "processed" / "p3_perfiles.parquet"
    crear_viajes_de_prueba().to_parquet(viajes_path)
    pd.DataFrame({"id": [1, 2]}).to_csv(estaciones_path, index=False)

    p3(viajes_path=viajes_path, estaciones_path=estaciones_path, output_path=output_path)

    assert output_path.exists()
    tabla = pd.read_parquet(output_path)
    assert set(tabla["conjunto"]) == set(PERIODOS)
    assert len(tabla) == 2 * len(PERIODOS)  # cada estación, una vez por periodo
    assert not tabla.duplicated(["estacion", "conjunto"]).any()
    esperadas = columnas_perfil("sal") + columnas_perfil("lle")
    assert set(esperadas + ["salidas_dia", "llegadas_dia", "balance"]) <= set(tabla.columns)
    assert tabla[esperadas].notna().all().all()

@pytest.mark.parametrize(
    ("hora", "franja"),
    [("00:00", 0), ("05:59", 0), ("06:00", 1), ("11:59", 1), ("12:00", 2), ("18:00", 3), ("23:59", 3)],
)
def test_franja_del_dia_en_los_limites(hora, franja):
    momentos = pd.Series(pd.to_datetime([f"2024-03-04 {hora}"]))
    assert franja_del_dia(momentos)[0] == franja

def test_perfil_suma_uno_y_cae_en_la_columna_correcta():
    # 2024-03-04 es lunes.
    momentos = pd.Series(pd.to_datetime(["2024-03-04 07:00", "2024-03-04 07:30", "2024-03-10 22:00"]))
    perfil = perfil_de_uso(pd.Series([1, 1, 1]), momentos, "sal")
    assert list(perfil.columns) == columnas_perfil("sal")
    np.testing.assert_allclose(perfil.sum(axis=1), 1.0)
    assert perfil.loc[1, "sal_lun_manana"] == pytest.approx(2 / 3)
    assert perfil.loc[1, "sal_dom_noche"] == pytest.approx(1 / 3)

def viajes_ida_y_vuelta(estacion_a, estacion_b, inicio, n):
    """n viajes de a → b, uno por hora desde `inicio`, cada uno de 20 minutos."""
    inicios = pd.date_range(inicio, periods=n, freq="h")
    return pd.DataFrame(
        {
            "inicio": inicios,
            "fin": inicios + pd.Timedelta(minutes=20),
            "origen_id": estacion_a,
            "destino_id": estacion_b,
        }
    )

def test_viajes_fuera_del_periodo_no_cambian_el_perfil():
    """El periodo es [desde, hasta): lo que pasa desde `hasta` no afecta (sin fuga a prueba)."""
    base = pd.concat([viajes_ida_y_vuelta(1, 2, "2024-03-04", 10), viajes_ida_y_vuelta(2, 1, "2024-03-05", 10)])
    despues = viajes_ida_y_vuelta(1, 2, "2024-03-11", 500)  # justo desde `hasta` en adelante
    args = dict(estaciones=np.array([1, 2]), desde="2024-03-04", hasta="2024-03-11", min_viajes=5)
    pd.testing.assert_frame_equal(
        perfiles_estaciones(base, **args),
        perfiles_estaciones(pd.concat([base, despues]), **args),
    )


def test_p3_ampliado_entrena_con_la_ventana_de_entrenamiento_y_validacion(tmp_path):
    viajes_path = tmp_path / "viajes.parquet"
    estaciones_path = tmp_path / "estaciones.csv"
    output_path = tmp_path / "p3_perfiles_ampliado.parquet"
    viajes = crear_viajes_de_prueba()
    viajes.to_parquet(viajes_path)
    pd.DataFrame({"id": [1, 2]}).to_csv(estaciones_path, index=False)
    p3_ampliado(viajes_path=viajes_path, estaciones_path=estaciones_path, output_path=output_path)
    tabla = pd.read_parquet(output_path)
    assert set(tabla["conjunto"]) == set(PERIODOS)
    esperado = perfiles_estaciones(viajes, np.array([1, 2]), FECHA_INICIO, FECHA_INICIO_PRUEBA)
    columnas = columnas_perfil("sal")
    ampliado = tabla[tabla["conjunto"] == "entrenamiento"].set_index("estacion")
    pd.testing.assert_frame_equal(
        ampliado[columnas].sort_index(), esperado[columnas].sort_index(), check_names=False
    )
    # `prueba` queda igual que en `p3`, para comparar las dos versiones sobre los mismos perfiles.
    prueba = perfiles_estaciones(viajes, np.array([1, 2]), FECHA_INICIO_PRUEBA, "2026-10-01")
    de_prueba = tabla[tabla["conjunto"] == "prueba"].set_index("estacion")
    pd.testing.assert_frame_equal(
        de_prueba[columnas].sort_index(), prueba[columnas].sort_index(), check_names=False
    )
