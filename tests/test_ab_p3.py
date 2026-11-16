"""Pruebas del A/B de P3: regla de decisión y métricas de los grupos."""

import numpy as np
import pandas as pd

from mibici_mlops.modeling.ab_p3 import gana, metricas_de_grupos

BASE = {"silhouette": 0.25, "davies_bouldin": 1.30, "calinski_harabasz": 160.0, "grupos": 3}


def alternativa(**cambios):
    return {**BASE, **cambios}


def test_gana_si_mejora_silhouette_mas_que_el_margen_y_davies_bouldin_no_empeora():
    ok, _ = gana(BASE, alternativa(silhouette=0.27, davies_bouldin=1.25), margen=0.01)
    assert ok


def test_no_gana_si_la_mejora_es_menor_que_el_margen():
    ok, motivo = gana(BASE, alternativa(silhouette=0.255, davies_bouldin=1.20), margen=0.01)
    assert not ok
    assert "margen" in motivo


def test_no_gana_si_davies_bouldin_empeora_aunque_el_silhouette_mejore():
    ok, motivo = gana(BASE, alternativa(silhouette=0.30, davies_bouldin=1.40), margen=0.01)
    assert not ok
    assert "Davies-Bouldin" in motivo


def test_no_gana_con_distinto_numero_de_grupos():
    ok, motivo = gana(BASE, alternativa(silhouette=0.40, grupos=4), margen=0.01)
    assert not ok
    assert "grupos" in motivo


def test_metricas_de_grupos_cuenta_los_grupos():
    rng = np.random.default_rng(0)
    perfil = pd.DataFrame(
        np.vstack([rng.normal(0, 0.1, (20, 3)), rng.normal(5, 0.1, (20, 3))])
    )
    etiquetas = np.repeat([0, 1], 20)
    metricas = metricas_de_grupos(perfil, etiquetas)
    assert metricas["grupos"] == 2
    assert metricas["silhouette"] > 0.9
