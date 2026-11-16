"""Pruebas del monitoreo de P2 con Evidently: preparación de columnas, drift por variable y por
mes, calidad de los datos y desempeño del modelo contra el baseline de frecuencia.

No necesitan el servidor de MLflow ni data/: usan una tabla simulada con las columnas de
`features p2` y un modelo doble cuyas probabilidades son las frecuencias de las semanas
anteriores, es decir, el baseline de frecuencia. Las reglas están en references/decisiones_p2.md,
decisión 12.
"""

from evidently import Report
import numpy as np
import pandas as pd

from mibici_mlops.features import CLASES_P2, DIAS_SEMANA
from mibici_mlops.modeling.train_p2 import predecir, probabilidades_modelo
from mibici_mlops.monitoreo import (
    CATEGORICAS_P2,
    COLUMNAS_MONITOREO_P2,
    PREDICCION,
    UMBRAL_DRIFT,
    calidad_p2,
    dataset_p2,
    desempeno_p2,
    desempeno_por_mes_p2,
    drift_por_mes_p2,
    graficar_desempeno_p2,
    preparar_p2,
    preset_drift,
    proporcion_clases,
    puntajes_drift,
    resumen_p2,
)

SEMILLA = 7


class ModeloDoble:
    """Da como probabilidad de cada clase su frecuencia en las semanas anteriores."""

    classes_ = np.array(["se_llena", "se_vacia", "sin_riesgo"])

    def predict_proba(self, X):
        vacia, llena = X["frec_se_vacia"].to_numpy(), X["frec_se_llena"].to_numpy()
        return np.column_stack([llena, vacia, 1 - vacia - llena])


def ventanas(
    desde: str,
    dias: int,
    estaciones: list[int],
    conjunto: str = "entrenamiento",
    riesgo: float = 0.2,
    semilla: int = SEMILLA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla simulada con las columnas de features p2, la predicción y sus probabilidades.

    `riesgo` es la fracción de ventanas en cada clase de riesgo. La frecuencia de las semanas
    anteriores suele ser más alta en las ventanas que sí tienen ese riesgo.
    """
    rng = np.random.default_rng(semilla)
    momentos = pd.date_range(f"{desde} 04:00", periods=dias * 12, freq="2h")
    tabla = pd.MultiIndex.from_product(
        [estaciones, momentos], names=["estacion", "ventana"]
    ).to_frame(index=False)
    n = len(tabla)
    tabla["dia"] = (tabla["ventana"] - pd.Timedelta(hours=4)).dt.normalize()
    tabla["hora"] = tabla["ventana"].dt.hour
    tabla["dia_semana"] = tabla["ventana"].dt.dayofweek
    tabla["feriado"] = False
    tabla["capacidad"] = 10 + tabla["estacion"]
    azar = rng.random(n)
    tabla["clase"] = np.where(
        azar < riesgo, "se_vacia", np.where(azar > 1 - riesgo, "se_llena", "sin_riesgo")
    )
    for clase in ["se_vacia", "se_llena"]:
        tabla[f"frec_{clase}"] = np.where(
            tabla["clase"] == clase,
            rng.choice([0.25, 0.5], n),
            rng.choice([0.0, 0.25], n, p=[0.8, 0.2]),
        )
    tabla["frec_sin_riesgo"] = 1 - tabla["frec_se_vacia"] - tabla["frec_se_llena"]
    probabilidades = probabilidades_modelo(ModeloDoble(), tabla)
    tabla[PREDICCION] = predecir(probabilidades)
    tabla["conjunto"] = conjunto
    return tabla, probabilidades


def drift(actual: pd.DataFrame, referencia: pd.DataFrame) -> pd.Series:
    reporte = Report([preset_drift(COLUMNAS_MONITOREO_P2)])
    return puntajes_drift(reporte.run(dataset_p2(actual), dataset_p2(referencia)))


# Preparación de las columnas


def test_preparar_p2_deja_las_columnas_monitoreadas_y_las_categoricas_como_texto():
    tabla, _ = ventanas("2025-01-06", 14, [1, 2, 3])
    datos = preparar_p2(tabla)
    assert list(datos.columns) == ["estacion", "ventana", *COLUMNAS_MONITOREO_P2]
    for columna in CATEGORICAS_P2:
        assert datos[columna].map(type).eq(str).all()
    assert set(datos["hora"]) == {f"{hora:02d}:00" for hora in range(0, 24, 2)}
    assert set(datos["dia_semana"]) == set(DIAS_SEMANA)
    assert set(datos["feriado"]) == {"no"}
    # La tabla original no cambia.
    assert tabla["hora"].dtype != object and tabla["feriado"].dtype == bool


# Drift


def test_sin_cambio_no_hay_drift_y_con_menos_riesgo_si():
    referencia, _ = ventanas("2025-01-06", 28, [1, 2, 3, 4])
    igual, _ = ventanas("2025-02-03", 28, [1, 2, 3, 4], semilla=SEMILLA + 1)
    menos_riesgo, _ = ventanas("2025-02-03", 28, [1, 2, 3, 4], riesgo=0.05, semilla=SEMILLA + 1)
    sin_cambio = drift(igual, referencia)
    con_cambio = drift(menos_riesgo, referencia)
    assert set(sin_cambio.index) == set(COLUMNAS_MONITOREO_P2)
    assert sin_cambio.lt(UMBRAL_DRIFT).all()
    assert con_cambio[["frec_se_vacia", "frec_se_llena", "clase"]].ge(UMBRAL_DRIFT).all()
    # El calendario no cambia: las mismas horas y los mismos días de la semana.
    assert con_cambio[["hora", "dia_semana"]].lt(UMBRAL_DRIFT).all()


def test_drift_por_mes_p2_compara_cada_mes_contra_el_mismo_mes_de_la_referencia():
    referencia, _ = ventanas("2024-01-01", 91, [1, 2, 3])
    actual, _ = ventanas("2025-01-01", 90, [1, 2, 3], conjunto="validacion", semilla=SEMILLA + 1)
    por_mes = drift_por_mes_p2(referencia, actual)
    assert list(por_mes.columns) == ["2025-01", "2025-02", "2025-03"]
    assert list(por_mes.index) == COLUMNAS_MONITOREO_P2
    assert por_mes.notna().all().all()
    assert por_mes.lt(UMBRAL_DRIFT).all().all()


# Calidad de los datos


def test_calidad_p2_da_cero_en_una_tabla_correcta():
    tabla, _ = ventanas("2025-01-06", 7, [1, 2])
    assert set(calidad_p2(tabla).values()) == {0}


def test_calidad_p2_cuenta_cada_tipo_de_problema():
    tabla, _ = ventanas("2025-01-06", 7, [1, 2])
    tabla = pd.concat([tabla, tabla.iloc[[0]]], ignore_index=True)  # una ventana repetida
    tabla.loc[1, "frec_se_vacia"] = np.nan
    tabla.loc[2, "frec_se_llena"] = 1.5
    tabla.loc[3, "hora"] = 3  # las ventanas de 2 horas empiezan en horas pares
    tabla.loc[4, "capacidad"] = 0
    tabla.loc[5, "clase"] = "otra"
    assert calidad_p2(tabla) == {
        "valores_faltantes": 1,
        "ventanas_duplicadas": 1,
        "frecuencias_fuera_de_0_a_1": 1,
        "horas_fuera_de_la_rejilla": 1,
        "capacidad_no_positiva": 1,
        "clases_desconocidas": 1,
    }


def test_proporcion_clases_incluye_las_tres_clases_en_orden():
    proporciones = proporcion_clases(
        pd.Series(["se_vacia", "sin_riesgo", "sin_riesgo", "se_vacia"])
    )
    assert list(proporciones) == CLASES_P2
    assert proporciones == {"se_vacia": 0.5, "se_llena": 0.0, "sin_riesgo": 0.5}


# Desempeño del modelo


def test_desempeno_p2_mide_al_modelo_y_al_baseline_de_frecuencia():
    """El modelo doble es el baseline de frecuencia: su ventaja sobre el baseline es 0."""
    tabla, probabilidades = ventanas("2025-01-06", 28, [1, 2, 3, 4])
    metricas = desempeno_p2(tabla, probabilidades)
    for riesgo in ["se_vacia", "se_llena"]:
        assert {f"recall_{riesgo}", f"precision_{riesgo}", f"pr_auc_{riesgo}"} <= set(metricas)
        assert metricas[f"proporcion_{riesgo}"] == round((tabla["clase"] == riesgo).mean(), 4)
    assert metricas["pr_auc_promedio"] == metricas["pr_auc_promedio_baseline"]
    assert metricas["mejora_sobre_baseline"] == 0
    # Las frecuencias sí distinguen el riesgo: la PR-AUC queda arriba de la del azar.
    assert metricas["pr_auc_se_vacia"] > 2 * metricas["proporcion_se_vacia"]


def test_desempeno_p2_solo_usa_las_filas_del_conjunto():
    tabla, probabilidades = ventanas("2025-01-06", 28, [1, 2, 3, 4])
    mitad = tabla[tabla["estacion"] <= 2]
    assert desempeno_p2(mitad, probabilidades) == desempeno_p2(
        mitad, probabilidades.loc[mitad.index]
    )


def test_desempeno_p2_es_none_si_falta_una_clase_de_riesgo():
    tabla, probabilidades = ventanas("2025-01-06", 7, [1, 2])
    sin_llenas = tabla[tabla["clase"] != "se_llena"]
    assert desempeno_p2(sin_llenas, probabilidades) is None


def test_desempeno_por_mes_p2_tiene_una_fila_por_mes():
    enero, prob_enero = ventanas("2025-01-01", 31, [1, 2], conjunto="validacion")
    febrero, prob_febrero = ventanas("2025-02-01", 28, [1, 2], conjunto="prueba")
    tabla = pd.concat([enero, febrero], ignore_index=True)
    probabilidades = pd.concat([prob_enero, prob_febrero], ignore_index=True)
    por_mes = desempeno_por_mes_p2(tabla, probabilidades)
    assert list(por_mes["mes"]) == ["2025-01", "2025-02"]
    assert list(por_mes["conjunto"]) == ["validacion", "prueba"]
    assert list(por_mes["ventanas"]) == [len(enero), len(febrero)]
    assert por_mes["mejora_sobre_baseline"].eq(0).all()


def test_graficar_desempeno_p2_tiene_un_panel_de_desempeno_y_otro_de_riesgo():
    enero, prob_enero = ventanas("2025-01-01", 31, [1, 2], conjunto="validacion")
    febrero, prob_febrero = ventanas("2025-02-01", 28, [1, 2], conjunto="prueba", riesgo=0.1)
    por_mes = desempeno_por_mes_p2(
        pd.concat([enero, febrero], ignore_index=True),
        pd.concat([prob_enero, prob_febrero], ignore_index=True),
    )
    arriba, abajo = graficar_desempeno_p2(por_mes, version="7").axes
    assert [texto.get_text() for texto in arriba.get_legend().get_texts()] == [
        "Champion (versión 7)",
        "Baseline de frecuencia",
    ]
    assert [texto.get_text() for texto in abajo.get_legend().get_texts()] == [
        "Se vacía",
        "Se llena",
    ]
    assert [marca.get_text() for marca in abajo.get_xticklabels()] == ["ene 25", "feb 25"]


# Resumen


def test_resumen_p2_cuenta_estaciones_nuevas_y_variables_con_drift():
    referencia, _ = ventanas("2025-01-06", 7, [1, 2, 3, 4, 5])
    actual, _ = ventanas("2025-02-03", 7, [3, 4, 5, 6, 7], conjunto="prueba")
    puntajes = pd.Series({"frec_se_vacia": 0.3, "capacidad": 0.05, PREDICCION: 0.02})
    datos = resumen_p2(referencia, actual, puntajes)
    assert datos["estaciones_nuevas"] == 2
    assert datos["ventanas_de_estaciones_nuevas"] == 2 * 7 * 12
    assert datos["estaciones_que_salen"] == 2
    assert datos["variables_con_drift"] == ["frec_se_vacia"]
    assert datos["proporcion_con_drift"] == 0.333
    assert datos["drift_del_conjunto"] is False
    assert datos["drift_de_la_prediccion"] == 0.02
