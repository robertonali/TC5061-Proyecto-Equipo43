"""Pruebas del entrenamiento de P1: hiperparámetros, predicción, métricas y reproducibilidad."""

import numpy as np
import pandas as pd
import pytest
import typer

from mibici_mlops.config import P1_BUSQUEDA_HGB, P1_HIPERPARAMETROS, SEMILLA
from mibici_mlops.features import COLUMNAS_P1
from mibici_mlops.modeling.train_p1 import (
    BUSQUEDA,
    MODELOS,
    crear_hgb,
    error_por_dia_semana,
    error_por_nivel_uso,
    metricas,
    nombre_run,
    predecir,
    resolver_hiperparametros,
)

N_FILAS = 2000
# Versiones pequeñas de cada candidato, para que las pruebas corran en segundos.
RAPIDOS = {
    "ridge": {"alpha": 1.0},
    "rf": {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 5, "max_samples": 0.5},
    "hgb": {"loss": "poisson", "learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 15},
}


def tabla_simulada(semilla=0):
    """Filas estación-día con las columnas de features.py; los viajes dependen del historial."""
    rng = np.random.default_rng(semilla)
    dia_semana = rng.integers(0, 7, N_FILAS)
    media_7d = rng.gamma(2, 10, N_FILAS)
    datos = pd.DataFrame(
        {
            "estacion_id": rng.integers(1, 30, N_FILAS),
            "dia_semana": dia_semana,
            "mes": rng.integers(1, 13, N_FILAS),
            "es_fin_de_semana": (dia_semana >= 5).astype(int),
            "es_feriado": (rng.random(N_FILAS) < 0.03).astype(int),
            "capacidad": rng.choice([12, 15, 20, 25], N_FILAS).astype(float),
            "lat": 20.6 + rng.normal(0, 0.02, N_FILAS),
            "lon": -103.3 + rng.normal(0, 0.02, N_FILAS),
            "lag_7d": rng.poisson(media_7d),
            "media_7d": media_7d,
        }
    )
    datos["viajes"] = rng.poisson(media_7d * np.where(dia_semana >= 5, 0.6, 1.1))
    return datos


def entrenar(modelo, datos):
    return MODELOS[modelo]["crear"](**RAPIDOS[modelo]).fit(datos[COLUMNAS_P1], datos["viajes"])


# resolver_hiperparametros


def test_hiperparametros_por_defecto():
    for modelo in ["ridge", "rf", "hgb"]:
        assert resolver_hiperparametros(modelo, {}) == P1_HIPERPARAMETROS[modelo]


def test_hiperparametros_de_la_terminal_reemplazan_los_de_config():
    # Las opciones que no se pasaron llegan como None y no reemplazan el valor por defecto.
    resultado = resolver_hiperparametros("rf", {"max_depth": 8, "n_estimators": None})
    assert resultado == {**P1_HIPERPARAMETROS["rf"], "max_depth": 8}


def test_hiperparametro_de_otro_modelo_falla():
    with pytest.raises(typer.BadParameter, match="alpha"):
        resolver_hiperparametros("hgb", {"alpha": 0.5})


def test_baseline_no_acepta_hiperparametros():
    assert resolver_hiperparametros("baseline", {"loss": None}) == {}
    with pytest.raises(typer.BadParameter, match="learning_rate"):
        resolver_hiperparametros("baseline", {"learning_rate": 0.05})


def test_busqueda_no_acepta_los_hiperparametros_de_la_rejilla():
    fijos = resolver_hiperparametros(BUSQUEDA, {})
    assert set(fijos).isdisjoint(P1_BUSQUEDA_HGB)
    assert resolver_hiperparametros(BUSQUEDA, {"max_iter": 100})["max_iter"] == 100
    with pytest.raises(typer.BadParameter, match="max_leaf_nodes"):
        resolver_hiperparametros(BUSQUEDA, {"max_leaf_nodes": 31})


def test_el_modelo_por_defecto_esta_en_la_rejilla_de_busqueda():
    for clave, valores in P1_BUSQUEDA_HGB.items():
        assert P1_HIPERPARAMETROS["hgb"][clave] in valores


# nombre_run


def test_nombre_run_distingue_las_combinaciones_de_la_rejilla():
    combinaciones = [
        {**P1_HIPERPARAMETROS["hgb"], "loss": loss, "max_leaf_nodes": hojas}
        for loss in P1_BUSQUEDA_HGB["loss"]
        for hojas in P1_BUSQUEDA_HGB["max_leaf_nodes"]
    ]
    nombres = {nombre_run("hgb", combinacion) for combinacion in combinaciones}
    assert len(nombres) == len(combinaciones)


def test_nombre_run_de_cada_modelo():
    assert nombre_run("baseline", {}) == "p1-baseline-semana-previa"
    assert nombre_run("ridge", P1_HIPERPARAMETROS["ridge"]) == "p1-ridge-alpha1.0"
    assert nombre_run("rf", P1_HIPERPARAMETROS["rf"]) == "p1-rf-arboles100-prof16-hoja20"
    assert (
        nombre_run("hgb", P1_HIPERPARAMETROS["hgb"]) == "p1-hgb-poisson-lr0.1-hojas63-arboles300"
    )


# predecir y metricas


def test_prediccion_no_es_negativa():
    """Ridge puede predecir valores negativos; predecir() los acota en 0."""
    entrenamiento = tabla_simulada(semilla=1)
    entrenamiento["viajes"] = 0
    entrenamiento.loc[entrenamiento["dia_semana"] == 0, "viajes"] = 50
    modelo = entrenar("ridge", entrenamiento)
    evaluacion = tabla_simulada(semilla=2)
    assert (modelo.predict(evaluacion[COLUMNAS_P1]) < 0).any()  # sin el recorte sí habría
    assert (predecir(modelo, evaluacion) >= 0).all()


def test_prediccion_usa_solo_las_columnas_del_modelo():
    """Columnas extra (como estacion_id o viajes) no cambian la predicción."""
    evaluacion = tabla_simulada(semilla=2)
    modelo = entrenar("hgb", tabla_simulada(semilla=1))
    np.testing.assert_array_equal(
        predecir(modelo, evaluacion), predecir(modelo, evaluacion[COLUMNAS_P1])
    )


def test_ridge_tolera_categorias_nuevas():
    """Un mes que no estaba en el entrenamiento no rompe la predicción."""
    entrenamiento = tabla_simulada(semilla=1)
    entrenamiento["mes"] = 1
    prediccion = predecir(entrenar("ridge", entrenamiento), tabla_simulada(semilla=2))
    assert np.isfinite(prediccion).all()


def test_metricas_de_una_prediccion_perfecta_y_con_error():
    real = pd.Series([0, 3, 10])
    assert metricas(real, real.to_numpy(), "val") == {"val_mae": 0.0, "val_rmse": 0.0}
    assert metricas(real, np.array([1, 3, 13]), "test") == pytest.approx(
        {"test_mae": 4 / 3, "test_rmse": np.sqrt(10 / 3)}
    )


# Tablas de error


def test_error_por_dia_semana():
    prueba = pd.DataFrame({"dia_semana": [0, 0, 6], "viajes": [10, 20, 5]})
    tabla = error_por_dia_semana(prueba, {"modelo": np.array([12, 20, 0])})
    assert tabla.set_index("dia_semana")["modelo"].to_dict() == {0: 1.0, 6: 5.0}


def test_error_por_nivel_uso_separa_estaciones_sin_historial():
    entrenamiento = pd.DataFrame({"estacion_id": [1, 2, 3], "viajes": [1, 10, 100]})
    prueba = pd.DataFrame({"estacion_id": [1, 3, 99], "viajes": [2, 90, 7]})
    tabla = error_por_nivel_uso(entrenamiento, prueba, {"modelo": np.array([2, 100, 0])})
    por_nivel = tabla.set_index("nivel_uso")
    assert por_nivel.loc["bajo", "modelo"] == 0
    assert por_nivel.loc["alto", "modelo"] == 10
    assert por_nivel.loc["sin historial", "modelo"] == 7
    assert por_nivel["filas"].sum() == len(prueba)


# Reproducibilidad


@pytest.mark.parametrize("modelo", ["ridge", "rf", "hgb"])
def test_entrenamiento_reproducible(modelo):
    """Dos entrenamientos con la misma semilla y los mismos datos dan las mismas predicciones.

    Se compara con una tolerancia de 1e-12: RandomForest promedia los árboles en paralelo
    (n_jobs=-1) y el orden de la suma cambia el último decimal (~1e-14), no el modelo.
    """
    entrenamiento = tabla_simulada(semilla=1)
    evaluacion = tabla_simulada(semilla=2)

    predicciones = [predecir(entrenar(modelo, entrenamiento), evaluacion) for _ in range(2)]
    np.testing.assert_allclose(predicciones[0], predicciones[1], rtol=1e-12)
    resultados = [metricas(evaluacion["viajes"], pred, "val") for pred in predicciones]
    assert resultados[0] == pytest.approx(resultados[1], rel=1e-12)


def test_otra_semilla_cambia_el_modelo():
    """La semilla sí se usa: con otra, el random forest elige otras muestras y cambia."""
    entrenamiento = tabla_simulada(semilla=1)
    evaluacion = tabla_simulada(semilla=2)
    otro = MODELOS["rf"]["crear"](**RAPIDOS["rf"]).set_params(modelo__random_state=SEMILLA + 1)
    otro.fit(entrenamiento[COLUMNAS_P1], entrenamiento["viajes"])
    assert not np.allclose(
        predecir(otro, evaluacion), predecir(entrenar("rf", entrenamiento), evaluacion)
    )


@pytest.mark.parametrize("modelo", ["rf", "hgb"])
def test_los_modelos_aleatorios_usan_la_semilla_del_proyecto(modelo):
    estimador = MODELOS[modelo]["crear"](**P1_HIPERPARAMETROS[modelo])[-1]
    assert estimador.random_state == SEMILLA


def test_hgb_sin_parada_anticipada():
    """Con parada anticipada, HGB separaría al azar una validación dentro del entrenamiento."""
    assert crear_hgb(**P1_HIPERPARAMETROS["hgb"])[-1].early_stopping is False


@pytest.mark.parametrize("modelo", ["rf", "hgb"])
def test_el_modelo_aprende_mas_que_el_baseline(modelo):
    entrenamiento = tabla_simulada(semilla=1)
    evaluacion = tabla_simulada(semilla=2)
    resultado = metricas(
        evaluacion["viajes"], predecir(entrenar(modelo, entrenamiento), evaluacion), "val"
    )
    baseline = metricas(evaluacion["viajes"], evaluacion["lag_7d"], "val")
    assert resultado["val_mae"] < baseline["val_mae"]
