"""Pruebas del A/B de P2: regla de decisión, bootstrap por días, métricas de cada versión y
linaje de la versión reentrenada.

No necesitan el servidor de MLflow: los runs son objetos simulados y las probabilidades se hacen a
mano. Las reglas están en references/decisiones_p2.md, decisión 13.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mibici_mlops.config import P2_AB_MARGEN_PR_AUC, P2_AB_TOLERANCIA_RECALL
from mibici_mlops.features import CLASES_P2
from mibici_mlops.modeling.ab_p2 import (
    comparar_por_mes,
    descripcion_del_run,
    diferencias_bootstrap,
    gana,
    intervalo_de_confianza,
    metricas_de_version,
)
from mibici_mlops.modeling.registrar_p2 import describir_desempeno, linaje

CHAMPION = {"pr_auc_promedio": 0.520, "recall_se_vacia": 0.71, "recall_se_llena": 0.79}


def alternativa(**cambios) -> dict:
    return {**CHAMPION, **cambios}


def evaluacion(dias: int = 20, por_dia: int = 60, semilla: int = 0) -> pd.DataFrame:
    """Ventanas simuladas con su día y su clase (20% en cada clase de riesgo)."""
    rng = np.random.default_rng(semilla)
    n = dias * por_dia
    azar = rng.random(n)
    return pd.DataFrame(
        {
            "dia": np.repeat(pd.date_range("2025-01-01", periods=dias, freq="D"), por_dia),
            "clase": np.where(
                azar < 0.2, "se_vacia", np.where(azar > 0.8, "se_llena", "sin_riesgo")
            ),
        }
    )


def probabilidades(datos: pd.DataFrame, acierto: float, semilla: int = 1) -> pd.DataFrame:
    """Probabilidades con ruido: con más `acierto`, la clase real recibe más probabilidad."""
    rng = np.random.default_rng(semilla)
    puntos = rng.random((len(datos), len(CLASES_P2)))
    for columna, clase in enumerate(CLASES_P2):
        puntos[:, columna] += acierto * (datos["clase"] == clase).to_numpy()
    return pd.DataFrame(puntos / puntos.sum(axis=1, keepdims=True), columns=CLASES_P2)


def run_simulado(metricas: dict, etapa: str, parametros: dict | None = None) -> SimpleNamespace:
    """Lo que `linaje` y `descripcion_del_run` leen de un run de MLflow."""
    return SimpleNamespace(
        info=SimpleNamespace(run_id="abc123", run_name="p2-hgb-de-prueba"),
        data=SimpleNamespace(
            tags={"commit": "0123456789", "stage_of_work": etapa, "model_family": "gradient"},
            params={"version_datos": "md5", **(parametros or {})},
            metrics=metricas,
        ),
    )


# Regla de decisión


def test_gana_si_mejora_mas_que_el_margen_con_intervalo_positivo_y_sin_perder_recall():
    ok, motivo = gana(CHAMPION, alternativa(pr_auc_promedio=0.530), (0.006, 0.014))
    assert ok
    assert "+0.0100" in motivo


def test_no_gana_si_la_mejora_es_menor_que_el_margen_aunque_sea_confiable():
    ok, motivo = gana(CHAMPION, alternativa(pr_auc_promedio=0.523), (0.002, 0.004))
    assert not ok
    assert "margen" in motivo


def test_no_gana_si_el_intervalo_de_confianza_incluye_el_cero():
    ok, motivo = gana(CHAMPION, alternativa(pr_auc_promedio=0.530), (-0.001, 0.021))
    assert not ok
    assert "intervalo" in motivo


@pytest.mark.parametrize("riesgo", ["se_vacia", "se_llena"])
def test_no_gana_si_baja_el_recall_de_una_clase_de_riesgo(riesgo):
    clave = f"recall_{riesgo}"
    peor = alternativa(pr_auc_promedio=0.530, **{clave: CHAMPION[clave] - 0.02})
    ok, motivo = gana(CHAMPION, peor, (0.006, 0.014))
    assert not ok
    assert f"recall de {riesgo}" in motivo


def test_gana_tolera_una_baja_pequena_del_recall():
    casi_igual = alternativa(pr_auc_promedio=0.530, recall_se_vacia=0.705)
    assert gana(CHAMPION, casi_igual, (0.006, 0.014))[0]


def test_la_regla_usa_por_defecto_los_valores_de_config():
    justo_abajo = alternativa(
        pr_auc_promedio=CHAMPION["pr_auc_promedio"] + P2_AB_MARGEN_PR_AUC / 2
    )
    assert not gana(CHAMPION, justo_abajo, (0.001, 0.004))[0]
    pierde_recall = alternativa(
        pr_auc_promedio=0.540,
        recall_se_llena=CHAMPION["recall_se_llena"] - 2 * P2_AB_TOLERANCIA_RECALL,
    )
    assert not gana(CHAMPION, pierde_recall, (0.015, 0.025))[0]


# Bootstrap por días


def test_intervalo_de_confianza_toma_la_parte_central_de_las_diferencias():
    assert intervalo_de_confianza(np.arange(101.0), confianza=0.9) == pytest.approx((5, 95))


def test_bootstrap_de_dos_versiones_iguales_da_diferencia_cero():
    datos = evaluacion()
    misma = probabilidades(datos, acierto=1.0)
    diferencias = diferencias_bootstrap(datos["dia"], datos["clase"], misma, misma, replicas=20)
    assert diferencias.shape == (20,)
    assert (diferencias == 0).all()


def test_bootstrap_es_reproducible_con_la_misma_semilla():
    datos = evaluacion()
    champion, mejor = probabilidades(datos, 0.5), probabilidades(datos, 1.5, semilla=2)
    argumentos = (datos["dia"], datos["clase"], champion, mejor)
    primera = diferencias_bootstrap(*argumentos, replicas=15, semilla=43)
    segunda = diferencias_bootstrap(*argumentos, replicas=15, semilla=43)
    otra_semilla = diferencias_bootstrap(*argumentos, replicas=15, semilla=44)
    np.testing.assert_array_equal(primera, segunda)
    assert not np.array_equal(primera, otra_semilla)


def test_bootstrap_de_una_version_claramente_mejor_queda_arriba_de_cero():
    datos = evaluacion()
    champion, mejor = probabilidades(datos, 0.5), probabilidades(datos, 1.5, semilla=2)
    diferencias = diferencias_bootstrap(datos["dia"], datos["clase"], champion, mejor, replicas=40)
    inferior, superior = intervalo_de_confianza(diferencias)
    real = (
        metricas_de_version(datos["clase"], mejor)["pr_auc_promedio"]
        - metricas_de_version(datos["clase"], champion)["pr_auc_promedio"]
    )
    assert inferior > 0
    assert inferior < real < superior


def test_bootstrap_con_un_solo_dia_repite_la_diferencia_real():
    """Si solo hay un día, todas las réplicas eligen ese día: no hay variación entre ellas."""
    datos = evaluacion(dias=1, por_dia=400)
    champion, mejor = probabilidades(datos, 0.5), probabilidades(datos, 1.5, semilla=2)
    diferencias = diferencias_bootstrap(datos["dia"], datos["clase"], champion, mejor, replicas=5)
    real = (
        metricas_de_version(datos["clase"], mejor)["pr_auc_promedio"]
        - metricas_de_version(datos["clase"], champion)["pr_auc_promedio"]
    )
    np.testing.assert_allclose(diferencias, real)


# Métricas de cada versión


def test_metricas_de_version_son_las_del_entrenamiento_sin_prefijo():
    datos = evaluacion()
    metricas = metricas_de_version(datos["clase"], probabilidades(datos, acierto=5.0))
    assert set(metricas) == {
        "recall_se_vacia",
        "precision_se_vacia",
        "pr_auc_se_vacia",
        "recall_se_llena",
        "precision_se_llena",
        "pr_auc_se_llena",
        "pr_auc_promedio",
    }
    assert metricas["pr_auc_promedio"] == pytest.approx(
        (metricas["pr_auc_se_vacia"] + metricas["pr_auc_se_llena"]) / 2
    )
    assert metricas["pr_auc_promedio"] > 0.95


def test_comparar_por_mes_da_la_diferencia_de_cada_mes():
    datos = evaluacion(dias=59)  # enero y febrero de 2025
    versiones = {
        "champion": probabilidades(datos, 0.5),
        "alternativa": probabilidades(datos, 1.5, semilla=2),
    }
    tabla = comparar_por_mes(datos, versiones)
    assert list(tabla.index) == ["2025-01", "2025-02"]
    assert list(tabla.columns) == ["champion", "alternativa", "diferencia"]
    np.testing.assert_allclose(
        tabla["diferencia"], tabla["alternativa"] - tabla["champion"], atol=1e-4
    )
    assert (tabla["diferencia"] > 0).all()


# De qué run sale cada versión


def test_descripcion_del_run_reentrenado_usa_lo_que_registro_el_run():
    run = run_simulado(
        {},
        "retrain",
        {
            "ventanas_entrenamiento": "6056700",
            "entrenamiento_hasta": "2026-03-31",
            "modelo": "HistGradientBoostingClassifier",
            "learning_rate": "0.05",
            "semilla": "43",
        },
    )
    descripcion = descripcion_del_run(run)
    assert descripcion["entrenamiento"] == {"ventanas": 6056700, "hasta": "2026-03-31"}
    # Solo los parámetros que definen el modelo: la semilla y los datos no son hiperparámetros.
    assert descripcion["hiperparametros"] == {
        "modelo": "HistGradientBoostingClassifier",
        "learning_rate": "0.05",
    }


def test_descripcion_de_un_run_anterior_deduce_hasta_cuando_entreno():
    """Los runs de la entrega 2 no registraron `entrenamiento_hasta`: entrenaron hasta el día
    anterior al inicio de la validación."""
    run = run_simulado(
        {},
        "candidate",
        {"ventanas_entrenamiento": "5247312", "fecha_inicio_validacion": "2025-10-01"},
    )
    assert descripcion_del_run(run)["entrenamiento"] == {
        "ventanas": 5247312,
        "hasta": "2025-09-30",
    }


def test_linaje_de_un_candidato_lleva_validacion_y_prueba():
    etiquetas = linaje(
        run_simulado({"val_pr_auc_promedio": 0.571, "test_pr_auc_promedio": 0.522}, "candidate")
    )
    assert etiquetas["val_pr_auc_promedio"] == "0.5710"
    assert etiquetas["test_pr_auc_promedio"] == "0.5220"
    assert etiquetas["stage_of_work"] == "candidate"
    assert describir_desempeno(etiquetas) == "0.5710 en validación y 0.5220 en prueba"


def test_linaje_de_un_reentrenamiento_no_inventa_la_metrica_de_validacion():
    etiquetas = linaje(run_simulado({"test_pr_auc_promedio": 0.5251}, "retrain"))
    assert "val_pr_auc_promedio" not in etiquetas
    assert etiquetas["stage_of_work"] == "retrain"
    assert describir_desempeno(etiquetas) == (
        "0.5251 en prueba (la validación es parte de su entrenamiento)"
    )
