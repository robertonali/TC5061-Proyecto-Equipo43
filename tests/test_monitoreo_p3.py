"""Pruebas del monitoreo de P3 con Evidently: grupos asignados, drift por variable y salud del
modelo. No necesitan el servidor de MLflow: el modelo es un doble que asigna el grupo por el
signo de la primera variable."""

from evidently import Report
import numpy as np
import pandas as pd

from mibici_mlops.features import COLUMNAS_P3, DIAS_SEMANA
from mibici_mlops.monitoreo import (
    GRUPO,
    UMBRAL_DRIFT,
    asignar_grupo,
    dataset_p3,
    matriz_drift,
    preset_drift,
    proporcion_grupos,
    puntajes_drift,
    resumen_p3,
    salud_del_modelo,
)

SEMILLA = 7
N = 120


class ModeloDoble:
    """Grupo 1 si la primera variable es mayor que 0.5, y 0 si no."""

    def predict(self, X):
        return (X.iloc[:, 0].to_numpy() > 0.5).astype(int)


def perfiles(ids, desplazamiento=0.0):
    """Perfiles simulados: dos bloques de variables entre 0 y 1; el primero separa dos grupos."""
    rng = np.random.default_rng(SEMILLA)
    datos = pd.DataFrame(rng.uniform(0, 1, (len(ids), len(COLUMNAS_P3))), columns=COLUMNAS_P3)
    datos[COLUMNAS_P3[1:]] = datos[COLUMNAS_P3[1:]] + desplazamiento
    datos.insert(0, "estacion", list(ids))
    return datos


def test_asignar_grupo_agrega_el_grupo_como_texto_sin_cambiar_el_original():
    original = perfiles(range(N))
    con_grupo = asignar_grupo(original, ModeloDoble())
    assert GRUPO not in original.columns
    assert set(con_grupo[GRUPO]) <= {"0", "1"}
    assert con_grupo[GRUPO].map(type).eq(str).all()


def test_sin_cambio_no_hay_drift_y_con_desplazamiento_si():
    ref = asignar_grupo(perfiles(range(N)), ModeloDoble())
    igual = asignar_grupo(perfiles(range(N)), ModeloDoble())
    movido = asignar_grupo(perfiles(range(N), desplazamiento=1.0), ModeloDoble())
    columnas = [*COLUMNAS_P3, GRUPO]
    sin_cambio = puntajes_drift(
        Report([preset_drift(columnas)]).run(dataset_p3(igual), dataset_p3(ref))
    )
    con_cambio = puntajes_drift(
        Report([preset_drift(columnas)]).run(dataset_p3(movido), dataset_p3(ref))
    )
    assert sin_cambio.lt(UMBRAL_DRIFT).all()
    assert con_cambio[COLUMNAS_P3[1:]].ge(UMBRAL_DRIFT).all()
    assert GRUPO in con_cambio.index


def test_matriz_drift_es_dia_por_franja():
    puntajes = pd.Series({c: i / 100 for i, c in enumerate(COLUMNAS_P3)})
    tabla = matriz_drift(puntajes, "lle")
    assert list(tabla.index) == DIAS_SEMANA
    assert tabla.shape == (7, 4)
    assert tabla.loc["lun", "madrugada"] == puntajes["lle_lun_madrugada"]


def test_salud_del_modelo_y_proporcion_de_grupos():
    conjunto = asignar_grupo(perfiles(range(N)), ModeloDoble())
    salud = salud_del_modelo(conjunto)
    assert set(salud) == {"silhouette", "davies_bouldin"}
    assert proporcion_grupos(conjunto).sum() == 1.0
    un_solo_grupo = conjunto.assign(**{GRUPO: "0"})
    assert salud_del_modelo(un_solo_grupo) == {}


def test_resumen_cuenta_estaciones_nuevas_y_las_que_salen():
    ref = asignar_grupo(perfiles(range(N)), ModeloDoble())
    actual = asignar_grupo(perfiles(range(5, N + 3)), ModeloDoble())
    puntajes = pd.Series({COLUMNAS_P3[0]: 0.3, COLUMNAS_P3[1]: 0.05, GRUPO: 0.02})
    datos = resumen_p3(ref, actual, puntajes)
    assert datos["estaciones_nuevas"] == [N, N + 1, N + 2]
    assert datos["estaciones_que_salen"] == [0, 1, 2, 3, 4]
    assert datos["variables_con_drift"] == [COLUMNAS_P3[0]]
    assert datos["drift_del_grupo"] == 0.02
    assert datos["drift_del_conjunto"] is False
