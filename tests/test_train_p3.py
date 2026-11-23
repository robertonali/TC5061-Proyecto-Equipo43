"""Pruebas del entrenamiento de P3: reproducibilidad, sin fuga entre conjuntos y casos límite."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from mibici_mlops.features import columnas_perfil
from mibici_mlops.modeling.train_p3 import BASELINE, MODELOS, P3Trainer, crear_modelo

SEMILLA = 0
N_POR_GRUPO = 20
CONJUNTOS = ["entrenamiento", "validacion", "prueba"]
COLUMNAS = columnas_perfil("sal") + columnas_perfil("lle")


def datos_sinteticos(n_grupos=3, semilla=SEMILLA, ruido=0.002):
    """Perfiles con `n_grupos` patrones bien separados; las mismas estaciones en cada conjunto."""
    rng = np.random.default_rng(semilla)
    patrones = rng.dirichlet(np.ones(len(COLUMNAS)), size=n_grupos)
    grupo = np.repeat(np.arange(n_grupos), N_POR_GRUPO)
    ids = np.arange(1, len(grupo) + 1)
    partes = []
    for conjunto in CONJUNTOS:
        valores = patrones[grupo] + rng.normal(0, ruido, (len(grupo), len(COLUMNAS)))
        parte = pd.DataFrame(valores, columns=COLUMNAS)
        parte.insert(0, "estacion", ids)
        parte["conjunto"] = conjunto
        partes.append(parte)
    perfiles = pd.concat(partes, ignore_index=True)
    estaciones = pd.DataFrame(
        {"id": ids, "latitude": 20.6 + grupo * 0.05, "longitude": -103.3 + grupo * 0.05}
    )
    return perfiles, estaciones, grupo


def candidato(nombre, random_state=SEMILLA):
    """El mismo modelo que crea p3train (vía crear_modelo), no uno armado en la prueba."""
    modelo, _, _ = crear_modelo(nombre, random_state)
    return modelo


@pytest.mark.parametrize("nombre", [*MODELOS, BASELINE])
def test_crear_modelo_fija_la_semilla(nombre):
    """Si crear_modelo deja de pasar random_state, el entrenamiento ya no es reproducible."""
    modelo, parametros, _ = crear_modelo(nombre, random_state=123)
    semillas = {k: v for k, v in modelo.get_params().items() if k.endswith("random_state")}
    assert semillas, f"{nombre} no tiene ningún parámetro random_state"
    assert set(semillas.values()) == {123}, f"{nombre} no usa la semilla: {semillas}"
    assert parametros["random_state"] == 123  # lo que se registra en MLflow


@pytest.mark.parametrize("nombre", [*MODELOS, BASELINE])
def test_misma_semilla_mismas_etiquetas_y_metricas(nombre):
    perfiles, estaciones, _ = datos_sinteticos(ruido=0.02)  # con ruido, para que la semilla importe
    por_coordenadas = nombre == BASELINE
    a = P3Trainer(perfiles, estaciones, candidato(nombre), por_coordenadas=por_coordenadas)
    b = P3Trainer(perfiles, estaciones, candidato(nombre), por_coordenadas=por_coordenadas)
    asignaciones_a, metricas_a = a.fit_predict()
    asignaciones_b, metricas_b = b.fit_predict()
    pd.testing.assert_frame_equal(asignaciones_a, asignaciones_b)
    # Las etiquetas son idénticas; las métricas, salvo el último dígito: con varios núcleos (p. ej.
    # en GitHub Actions) KMeans suma la inercia en hilos y el orden de las sumas puede cambiar
    # entre corridas (1.2991691317088154 contra 1.2991691317088156).
    assert metricas_a == pytest.approx(metricas_b, rel=1e-12)


def test_recupera_grupos_bien_separados():
    perfiles, estaciones, grupo = datos_sinteticos(n_grupos=3)
    trainer = P3Trainer(perfiles, estaciones, candidato("kmeans_k3"))
    _, metricas = trainer.fit_predict()
    assert adjusted_rand_score(grupo, trainer.etiquetas) == pytest.approx(1.0)
    assert metricas["ari_estabilidad"] == pytest.approx(1.0)
    assert metricas["train_silhouette"] > 0.8


def test_validacion_y_prueba_no_cambian_el_modelo():
    """El modelo solo se ajusta con entrenamiento: alterar validación y prueba no lo mueve."""
    perfiles, estaciones, _ = datos_sinteticos()
    alterados = perfiles.copy()
    fuera = alterados["conjunto"] != "entrenamiento"
    alterados.loc[fuera, COLUMNAS] = alterados.loc[fuera, COLUMNAS].to_numpy()[::-1]
    original = P3Trainer(perfiles, estaciones, candidato("kmeans_k3"))
    modificado = P3Trainer(alterados, estaciones, candidato("kmeans_k3"))
    original.fit_predict()
    modificado.fit_predict()
    np.testing.assert_array_equal(original.etiquetas, modificado.etiquetas)
    np.testing.assert_allclose(
        original.modelo.cluster_centers_, modificado.modelo.cluster_centers_
    )


def test_asignaciones_una_fila_por_estacion_y_conjunto_con_coordenadas():
    perfiles, estaciones, _ = datos_sinteticos()
    asignaciones, _ = P3Trainer(perfiles, estaciones, candidato("kmeans_k3")).fit_predict()
    assert len(asignaciones) == len(perfiles)
    assert not asignaciones.duplicated(["estacion", "conjunto"]).any()
    assert set(asignaciones["cluster"]) <= set(range(3))
    assert asignaciones[["estacion_lat", "estacion_lon"]].notna().all().all()


def test_metricas_segun_la_familia():
    perfiles, estaciones, _ = datos_sinteticos()
    _, kmeans = P3Trainer(perfiles, estaciones, candidato("kmeans_k3")).fit_predict()
    _, gmm = P3Trainer(perfiles, estaciones, candidato("gmm_diag_k2")).fit_predict()
    assert "inertia" in kmeans and "bic" not in kmeans
    assert "bic" in gmm and "inertia" not in gmm
    for prefijo in ["train", "val", "test"]:
        assert f"{prefijo}_silhouette" in kmeans


def test_baseline_ajusta_con_coordenadas_y_evalua_con_el_perfil():
    perfiles, estaciones, _ = datos_sinteticos()
    trainer = P3Trainer(perfiles, estaciones, candidato(BASELINE), por_coordenadas=True)
    trainer.fit_predict()
    assert list(trainer.Z.columns) == ["estacion_lat", "estacion_lon"]
    assert list(trainer.X.columns) == COLUMNAS


def test_un_solo_grupo_falla_con_mensaje_claro():
    """Perfiles idénticos: KMeans deja un solo grupo y silhouette no se puede calcular."""
    perfiles, estaciones, _ = datos_sinteticos(n_grupos=1, ruido=0)
    with pytest.warns(Warning), pytest.raises(ValueError, match="al menos 2 clusters"):
        P3Trainer(perfiles, estaciones, candidato("kmeans_k3")).fit_predict()


def test_sin_estaciones_comunes_entre_entrenamiento_y_validacion_falla():
    perfiles, estaciones, _ = datos_sinteticos()
    validacion = perfiles["conjunto"] == "validacion"
    perfiles.loc[validacion, "estacion"] += 1000  # ids que no están en entrenamiento
    estaciones = pd.concat([estaciones, estaciones.assign(id=estaciones["id"] + 1000)])
    with pytest.raises(ValueError, match="estaciones comunes"):
        P3Trainer(perfiles, estaciones, candidato("kmeans_k3")).fit_predict()


def test_reentrenamiento_no_acepta_las_rutas_del_champion():
    """Sin rutas propias, el reentrenamiento pisaría el modelo y las asignaciones del champion."""
    import typer

    from mibici_mlops.modeling.train_p3 import p3train

    with pytest.raises(typer.BadParameter, match="archivos propios"):
        p3train(modelo="kmeans_k3", reentrenamiento=True)
