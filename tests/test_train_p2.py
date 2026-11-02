"""Pruebas del entrenamiento de P2: partición, baselines, hiperparámetros, predicción, métricas,
estabilidad mensual y reproducibilidad.

Usan una tabla simulada con las columnas de `features p2` y versiones pequeñas de los candidatos,
para que corran en segundos. Las reglas que se prueban están en references/decisiones_p2.md
(decisiones 6, 7 y 8).
"""

import numpy as np
import pandas as pd
import pytest
import typer

from mibici_mlops.config import (
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    P2_BUSQUEDA_HGB,
    P2_HIPERPARAMETROS,
    SEMILLA,
)
from mibici_mlops.features import CLASES_P2, COLUMNAS_P2
from mibici_mlops.modeling.train_p2 import (
    BUSQUEDA,
    MODELOS,
    VARIABLES,
    asignar_conjunto,
    baseline_clase_mayoritaria,
    baseline_frecuencia,
    calcular_metricas,
    crear_hgb,
    crear_logistica,
    entrenar,
    nombre_run,
    pesos_de_clase,
    pr_auc_por_mes,
    predecir,
    probabilidades_modelo,
    resolver_hiperparametros,
)

N_FILAS = 3000
# Versiones pequeñas de cada candidato, para que las pruebas corran en segundos.
RAPIDOS = {
    "logistica": {"c": 1.0, "max_iter": 500, "class_weight": "balanced"},
    "hgb": {
        "learning_rate": 0.1,
        "max_iter": 20,
        "max_leaf_nodes": 15,
        "class_weight": "balanced",
    },
}


def tabla_simulada(semilla: int = 0) -> pd.DataFrame:
    """Ventanas con las columnas de features.py; la clase depende de la frecuencia y la hora."""
    rng = np.random.default_rng(semilla)
    frec_se_vacia = rng.choice([0, 0.25, 0.5, 0.75, 1], N_FILAS)
    frec_se_llena = (1 - frec_se_vacia) * rng.choice([0, 0.5, 1], N_FILAS)
    hora = rng.choice(np.arange(0, 24, 2), N_FILAS)
    pico = np.isin(hora, [6, 8, 18])
    azar = rng.random(N_FILAS)
    clase = np.where(
        azar < 0.15 + 0.6 * frec_se_vacia * pico,
        "se_vacia",
        np.where(azar > 0.85 - 0.6 * frec_se_llena, "se_llena", "sin_riesgo"),
    )
    dia = pd.Timestamp("2025-01-06") + pd.to_timedelta(rng.integers(0, 90, N_FILAS), unit="D")
    return pd.DataFrame(
        {
            "estacion": rng.integers(1, 40, N_FILAS),
            "dia": dia,
            "hora": hora,
            "dia_semana": dia.dayofweek,
            "feriado": rng.random(N_FILAS) < 0.03,
            "capacidad": rng.choice([12, 15, 20, 25], N_FILAS),
            "clase": clase,
            "frec_se_vacia": frec_se_vacia,
            "frec_se_llena": frec_se_llena,
            "frec_sin_riesgo": 1 - frec_se_vacia - frec_se_llena,
        }
    )


def entrenado(modelo: str, datos: pd.DataFrame):
    return entrenar(MODELOS[modelo]["crear"](**RAPIDOS[modelo]), datos)[0]


def probabilidades(*filas: tuple[float, float, float]) -> pd.DataFrame:
    """Probabilidades (se_vacia, se_llena, sin_riesgo) por fila."""
    return pd.DataFrame(filas, columns=CLASES_P2)


# Variables


def test_variables_no_incluyen_la_etiqueta_ni_columnas_inexistentes():
    assert "clase" not in VARIABLES
    assert set(VARIABLES) <= set(COLUMNAS_P2)
    # frec_sin_riesgo es 1 menos las otras dos: sería redundante.
    assert "frec_sin_riesgo" not in VARIABLES


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


def test_asignar_conjunto_conserva_el_indice():
    dias = pd.Series(pd.to_datetime(["2023-01-01", "2026-05-01"]), index=[10, 20])
    assert list(asignar_conjunto(dias).index) == [10, 20]


# Baselines


def test_clase_mayoritaria_predice_siempre_la_mas_comun_del_entrenamiento():
    entrenamiento = pd.DataFrame({"clase": ["sin_riesgo"] * 8 + ["se_vacia", "se_llena"]})
    evaluacion = pd.DataFrame(index=[5, 6, 7])
    resultado = baseline_clase_mayoritaria(entrenamiento, evaluacion)
    assert list(resultado.columns) == CLASES_P2
    assert list(resultado.index) == [5, 6, 7]
    assert (predecir(resultado) == "sin_riesgo").all()


def test_clase_mayoritaria_aprende_de_los_datos_no_esta_fija():
    entrenamiento = pd.DataFrame({"clase": ["se_vacia"] * 3 + ["sin_riesgo", "se_llena"]})
    resultado = baseline_clase_mayoritaria(entrenamiento, pd.DataFrame(index=[0]))
    assert predecir(resultado).iloc[0] == "se_vacia"


@pytest.mark.xfail(
    raises=KeyError,
    strict=True,
    reason="Si el entrenamiento no tiene alguna de las 3 clases, el modelo no da su probabilidad "
    "y falla al ordenar las columnas. En el proyecto el entrenamiento tiene millones de ventanas "
    "con las 3 clases, así que no ocurre; lo esperado sería probabilidad 0 para esa clase.",
)
def test_entrenamiento_sin_alguna_clase():
    entrenamiento = pd.DataFrame({"clase": ["se_vacia"] * 3 + ["sin_riesgo"]})
    resultado = baseline_clase_mayoritaria(entrenamiento, pd.DataFrame(index=[0]))
    assert resultado.loc[0, "se_llena"] == 0


def test_baseline_frecuencia_usa_las_columnas_de_las_semanas_anteriores():
    evaluacion = tabla_simulada().head(5)
    resultado = baseline_frecuencia(pd.DataFrame(), evaluacion)
    assert list(resultado.columns) == CLASES_P2
    np.testing.assert_array_equal(resultado["se_vacia"], evaluacion["frec_se_vacia"])
    np.testing.assert_array_equal(resultado["sin_riesgo"], evaluacion["frec_sin_riesgo"])


# predecir


@pytest.mark.parametrize(
    "fila, esperada",
    [
        ((0.1, 0.2, 0.7), "sin_riesgo"),
        ((0.6, 0.3, 0.1), "se_vacia"),
        ((0.5, 0.0, 0.5), "se_vacia"),  # empate: gana el riesgo (decisión 7)
        ((0.0, 0.5, 0.5), "se_llena"),
        ((0.5, 0.5, 0.0), "se_vacia"),  # empate entre riesgos: "se vacía" va primero
        ((0.25, 0.25, 0.5), "sin_riesgo"),
    ],
)
def test_predecir_y_desempates(fila, esperada):
    assert predecir(probabilidades(fila)).iloc[0] == esperada


# pesos_de_clase


@pytest.mark.parametrize("opcion, esperado", [("balanced", "balanced"), ("none", None)])
def test_pesos_de_clase(opcion, esperado):
    assert pesos_de_clase(opcion) == esperado


@pytest.mark.parametrize("opcion", ["Balanced", "", "auto"])
def test_pesos_de_clase_rechaza_otras_opciones(opcion):
    with pytest.raises(typer.BadParameter):
        pesos_de_clase(opcion)


# resolver_hiperparametros


@pytest.mark.parametrize("modelo", ["logistica", "hgb"])
def test_sin_opciones_usa_los_valores_de_config(modelo):
    vacias = dict.fromkeys(["c", "max_iter", "class_weight", "learning_rate", "max_leaf_nodes"])
    assert resolver_hiperparametros(modelo, vacias) == P2_HIPERPARAMETROS[modelo]


def test_las_opciones_de_la_terminal_reemplazan_a_las_de_config():
    resultado = resolver_hiperparametros("hgb", {"learning_rate": 0.05, "c": None})
    assert resultado == {**P2_HIPERPARAMETROS["hgb"], "learning_rate": 0.05}


@pytest.mark.parametrize(
    "modelo, opciones",
    [
        ("logistica", {"learning_rate": 0.1}),  # es de hgb
        ("hgb", {"c": 2.0}),  # es de la logística
        ("mayoritaria", {"max_iter": 10}),  # los baselines no tienen hiperparámetros
        (BUSQUEDA, {"learning_rate": 0.1}),  # la rejilla define learning_rate
    ],
)
def test_hiperparametro_que_el_modelo_no_usa_da_error(modelo, opciones):
    """Un error de dedo no se ignora en silencio."""
    with pytest.raises(typer.BadParameter, match="no acepta"):
        resolver_hiperparametros(modelo, opciones)


def test_la_busqueda_solo_fija_lo_que_no_esta_en_la_rejilla():
    fijos = resolver_hiperparametros(BUSQUEDA, {})
    assert not set(fijos) & set(P2_BUSQUEDA_HGB)
    assert fijos["max_iter"] == P2_HIPERPARAMETROS["hgb"]["max_iter"]


# nombre_run


@pytest.mark.parametrize(
    "modelo, hiperparametros, esperado",
    [
        ("mayoritaria", {}, "p2-baseline-clase-mayoritaria"),
        ("frecuencia", {}, "p2-baseline-frecuencia-4-semanas"),
        ("logistica", {"c": 1.0, "class_weight": "balanced"}, "p2-logistica-balanceado-c1.0"),
        (
            "hgb",
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 300, "class_weight": "none"},
            "p2-hgb-sin-pesos-lr0.05-hojas31-arboles300",
        ),
        (
            BUSQUEDA,
            {"max_iter": 300, "class_weight": "balanced"},
            "p2-busqueda-hgb-balanceado-arboles300",
        ),
    ],
)
def test_nombre_run(modelo, hiperparametros, esperado):
    assert nombre_run(modelo, hiperparametros) == esperado


# Modelos


def test_hgb_usa_la_semilla_del_proyecto_y_no_para_antes():
    """Con parada anticipada, HGB separaría al azar una parte del entrenamiento y mezclaría
    fechas."""
    estimador = crear_hgb(**P2_HIPERPARAMETROS["hgb"])[-1]
    assert estimador.random_state == SEMILLA
    assert estimador.early_stopping is False
    assert list(estimador.categorical_features) == ["hora", "dia_semana"]


def test_logistica_sin_pesos_de_clase():
    assert crear_logistica(c=1.0, max_iter=100, class_weight="none")[-1].class_weight is None


@pytest.mark.parametrize("modelo", ["logistica", "hgb"])
def test_probabilidades_en_el_orden_de_las_clases_y_suman_1(modelo):
    datos = tabla_simulada()
    resultado = probabilidades_modelo(entrenado(modelo, datos), datos.head(50))
    assert list(resultado.columns) == CLASES_P2
    np.testing.assert_allclose(resultado.sum(axis=1), 1)


@pytest.mark.parametrize("modelo", ["logistica", "hgb"])
def test_el_modelo_aprende_mas_que_la_clase_mayoritaria(modelo):
    entrenamiento, evaluacion = tabla_simulada(semilla=1), tabla_simulada(semilla=2)
    clase = evaluacion["clase"]
    modelo_ = calcular_metricas(
        clase, probabilidades_modelo(entrenado(modelo, entrenamiento), evaluacion), "val"
    )
    piso = calcular_metricas(clase, baseline_clase_mayoritaria(entrenamiento, evaluacion), "val")
    assert modelo_["val_pr_auc_promedio"] > piso["val_pr_auc_promedio"] + 0.1


def test_el_modelo_funciona_con_una_categoria_que_no_vio():
    """Una hora que no estaba en el entrenamiento (p. ej. si cambia la ventana) no rompe la
    predicción de la logística: OneHotEncoder la ignora."""
    entrenamiento = tabla_simulada()
    entrenamiento = entrenamiento[entrenamiento["hora"] != 22]
    nueva = tabla_simulada().head(3).assign(hora=22)
    resultado = probabilidades_modelo(entrenado("logistica", entrenamiento), nueva)
    assert resultado.notna().all().all()


# calcular_metricas


def test_metricas_de_un_modelo_perfecto():
    clase = pd.Series(["se_vacia", "se_llena", "sin_riesgo", "sin_riesgo"])
    perfectas = probabilidades((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1))
    metricas = calcular_metricas(clase, perfectas, "test")
    for riesgo in ["se_vacia", "se_llena"]:
        assert metricas[f"test_recall_{riesgo}"] == 1
        assert metricas[f"test_precision_{riesgo}"] == 1
        assert metricas[f"test_pr_auc_{riesgo}"] == 1
        assert metricas[f"test_proporcion_{riesgo}"] == 0.25
    assert metricas["test_pr_auc_promedio"] == 1


def test_metricas_si_nunca_predice_riesgo():
    """Precisión 0 en lugar de error cuando no hay ninguna predicción de riesgo."""
    clase = pd.Series(["se_vacia", "se_llena", "sin_riesgo", "sin_riesgo"])
    siempre_sin_riesgo = probabilidades(*[(0, 0, 1)] * 4)
    metricas = calcular_metricas(clase, siempre_sin_riesgo, "val")
    assert metricas["val_recall_se_vacia"] == 0
    assert metricas["val_precision_se_vacia"] == 0
    # Sin distinguir nada, la PR-AUC es la proporción de la clase: la del azar.
    assert metricas["val_pr_auc_se_vacia"] == pytest.approx(metricas["val_proporcion_se_vacia"])


def test_el_promedio_es_la_media_de_las_dos_clases_de_riesgo():
    datos = tabla_simulada()
    metricas = calcular_metricas(datos["clase"], baseline_frecuencia(datos, datos), "val")
    assert metricas["val_pr_auc_promedio"] == pytest.approx(
        (metricas["val_pr_auc_se_vacia"] + metricas["val_pr_auc_se_llena"]) / 2
    )


# pr_auc_por_mes


def test_pr_auc_por_mes_una_fila_por_mes_y_conjunto():
    datos = tabla_simulada()
    conjuntos = {"val": datos.iloc[:1500], "test": datos.iloc[1500:]}
    tabla = pr_auc_por_mes(
        conjuntos, {nombre: baseline_frecuencia(d, d) for nombre, d in conjuntos.items()}
    )
    # tabla_simulada cubre de enero a principios de abril de 2025: 4 meses por conjunto.
    assert tabla.groupby("conjunto").size().to_dict() == {"test": 4, "val": 4}
    np.testing.assert_allclose(tabla["promedio"], tabla[["se_vacia", "se_llena"]].mean(axis=1))


# Reproducibilidad


@pytest.mark.parametrize("modelo", ["logistica", "hgb"])
def test_entrenamiento_reproducible(modelo):
    """Dos entrenamientos con la misma semilla y los mismos datos dan exactamente las mismas
    probabilidades, clases y métricas."""
    entrenamiento, evaluacion = tabla_simulada(semilla=1), tabla_simulada(semilla=2)
    resultados = [
        probabilidades_modelo(entrenado(modelo, entrenamiento), evaluacion) for _ in range(2)
    ]
    pd.testing.assert_frame_equal(resultados[0], resultados[1], check_exact=True)
    pd.testing.assert_series_equal(predecir(resultados[0]), predecir(resultados[1]))
    assert calcular_metricas(evaluacion["clase"], resultados[0], "val") == calcular_metricas(
        evaluacion["clase"], resultados[1], "val"
    )


@pytest.mark.parametrize("modelo", ["logistica", "hgb"])
def test_entrenamiento_no_depende_del_orden_de_las_filas(modelo):
    """Leer la tabla en otro orden no cambia el modelo (más que en el último decimal)."""
    entrenamiento, evaluacion = tabla_simulada(semilla=1), tabla_simulada(semilla=2)
    revuelto = entrenamiento.sample(frac=1, random_state=SEMILLA)
    original = probabilidades_modelo(entrenado(modelo, entrenamiento), evaluacion)
    otro_orden = probabilidades_modelo(entrenado(modelo, revuelto), evaluacion)
    np.testing.assert_allclose(original, otro_orden, atol=1e-6)
