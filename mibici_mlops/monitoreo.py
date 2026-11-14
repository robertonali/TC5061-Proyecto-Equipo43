"""Monitoreo de los datos de P1 con Evidently: drift y calidad de los datos.

Compara el conjunto de referencia (entrenamiento del modelo `champion`, 2022-03-08 a 2025-09-30)
con el conjunto actual (los 12 meses posteriores, P1_MONITOREO_INICIO a P1_MONITOREO_FIN de
config.py). Los periodos y el porqué están en references/decisiones_p1.md, decisión 8.

Lee data/processed/p1_estacion_dia.parquet (la salida de `features p1`) y guarda en
reports/evidently/:
- p1_drift_calidad.html: reporte de Evidently con el drift por variable (DataDriftPreset) y la
  calidad de los datos de ambos conjuntos (DataSummaryPreset).
- p1_drift_resumen.json: periodos, filas, estaciones nuevas y el puntaje de drift por variable.
- p1_drift_por_mes.csv y p1_drift_por_mes.png: el drift de cada mes del periodo actual contra el
  mismo mes de los años de entrenamiento, para ver si el cambio crece con el tiempo.

Se corre con el entorno del proyecto (.venv):

    python -m mibici_mlops.monitoreo p1
"""

import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import pandas as pd
import typer

from mibici_mlops.config import (
    EVIDENTLY_DIR,
    P1_FEATURES_PATH,
    P1_MONITOREO_FIN,
    P1_MONITOREO_INICIO,
)
from mibici_mlops.features import COLUMNAS_P1

app = typer.Typer()


@app.callback()
def main():
    """Monitoreo de los datos con Evidently. Un subcomando por pregunta."""


OBJETIVO = "viajes"
# Para Evidently, las variables de calendario son categóricas (también los indicadores binarios):
# su drift se mide con la distancia de Jensen-Shannon y no con la de Wasserstein.
CATEGORICAS = ["dia_semana", "mes", "es_fin_de_semana", "es_feriado"]
NUMERICAS = [OBJETIVO, *(c for c in COLUMNAS_P1 if c not in CATEGORICAS)]

# Métodos y umbral fijos (los mismos que Evidently elige por defecto con más de 1,000 filas), para
# que no cambien con el tamaño de la muestra: con cientos de miles de filas, una prueba de
# hipótesis (KS, chi cuadrada) marca como drift cualquier diferencia, por pequeña que sea. Una
# distancia mide cuánto cambió la distribución. Una variable tiene drift si la distancia >= 0.1;
# el conjunto, si la mitad o más de las columnas tienen drift.
METODO_NUMERICAS = "wasserstein"
METODO_CATEGORICAS = "jensenshannon"
UMBRAL_DRIFT = 0.1
PROPORCION_DRIFT = 0.5


def separar_periodos(
    tabla: pd.DataFrame, inicio: str = P1_MONITOREO_INICIO, fin: str = P1_MONITOREO_FIN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Referencia (filas de entrenamiento) y actual (filas con fecha entre `inicio` y `fin`)."""
    referencia = tabla[tabla["conjunto"].astype(str) == "entrenamiento"]
    actual = tabla[tabla["fecha"].between(pd.Timestamp(inicio), pd.Timestamp(fin))]
    return referencia, actual


def preparar(conjunto: pd.DataFrame) -> pd.DataFrame:
    """Columnas que se monitorean. Las categóricas pasan a texto: Evidently no acepta int8."""
    datos = conjunto[["estacion_id", "fecha", OBJETIVO, *COLUMNAS_P1]].copy()
    datos[CATEGORICAS] = datos[CATEGORICAS].astype(int).astype(str)
    return datos


def dataset(conjunto: pd.DataFrame) -> Dataset:
    definicion = DataDefinition(
        numerical_columns=NUMERICAS,
        categorical_columns=CATEGORICAS,
        datetime_columns=["fecha"],
        id_column="estacion_id",
    )
    return Dataset.from_pandas(preparar(conjunto), data_definition=definicion)


def preset_drift(columnas: list[str] | None = None) -> DataDriftPreset:
    return DataDriftPreset(
        columns=columnas,
        num_method=METODO_NUMERICAS,
        cat_method=METODO_CATEGORICAS,
        threshold=UMBRAL_DRIFT,
        drift_share=PROPORCION_DRIFT,
    )


def puntajes_drift(snapshot) -> pd.Series:
    """Distancia de drift por variable, leída de las métricas ValueDrift del reporte."""
    puntajes = {
        metrica["config"]["column"]: metrica["value"]
        for metrica in snapshot.dict()["metrics"]
        if metrica["config"]["type"].endswith(":ValueDrift")
    }
    return pd.Series(puntajes, name="drift").sort_values(ascending=False)


def drift_por_mes(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Drift de cada mes de `actual` contra el mismo mes del año en `referencia`.

    Comparar octubre de 2025 con los octubres de entrenamiento quita la estacionalidad: lo que
    queda es el cambio de un año a otro. `mes` se excluye porque es constante dentro de cada mes.
    Regresa una tabla variable × mes con la distancia de drift.
    """
    columnas = [c for c in [*NUMERICAS, *CATEGORICAS] if c != "mes"]
    resultados = {}
    for periodo, mes_actual in actual.groupby(actual["fecha"].dt.to_period("M")):
        mismo_mes = referencia[referencia["fecha"].dt.month == periodo.month]
        snapshot = Report([preset_drift(columnas)]).run(dataset(mes_actual), dataset(mismo_mes))
        resultados[str(periodo)] = puntajes_drift(snapshot)
    return pd.DataFrame(resultados).loc[columnas]


# Escala secuencial de un solo tono (azul, de claro a oscuro) para la magnitud del drift.
AZULES = LinearSegmentedColormap.from_list(
    "azules", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def graficar_drift_por_mes(por_mes: pd.DataFrame) -> Figure:
    """Mapa de calor variable × mes; las celdas con drift (>= UMBRAL_DRIFT) llevan el valor."""
    figura = Figure(figsize=(10, 4.8), layout="constrained")
    eje = figura.subplots()
    maximo = max(0.3, float(por_mes.to_numpy().max()))
    imagen = eje.imshow(por_mes.to_numpy(), cmap=AZULES, vmin=0, vmax=maximo, aspect="auto")
    eje.set_xticks(range(por_mes.shape[1]), por_mes.columns, rotation=45, ha="right")
    eje.set_yticks(range(por_mes.shape[0]), por_mes.index)
    for fila, variable in enumerate(por_mes.index):
        for columna, mes in enumerate(por_mes.columns):
            valor = por_mes.loc[variable, mes]
            if valor >= UMBRAL_DRIFT:
                color = "white" if valor > maximo * 0.55 else "#1f1f1f"
                eje.text(
                    columna,
                    fila,
                    f"{valor:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )
    eje.tick_params(length=0)
    for borde in eje.spines.values():
        borde.set_visible(False)
    barra = figura.colorbar(imagen, ax=eje, shrink=0.8)
    barra.set_label("Distancia de drift (Wasserstein normalizada / Jensen-Shannon)")
    barra.ax.axhline(UMBRAL_DRIFT, color="#1f1f1f", linewidth=1)
    eje.set_title(
        f"P1: drift de cada mes contra el mismo mes en entrenamiento "
        f"(con número: drift >= {UMBRAL_DRIFT})",
        loc="left",
        fontsize=11,
    )
    return figura


def describir(conjunto: pd.DataFrame) -> dict:
    return {
        "desde": f"{conjunto['fecha'].min():%Y-%m-%d}",
        "hasta": f"{conjunto['fecha'].max():%Y-%m-%d}",
        "filas": len(conjunto),
        "estaciones": int(conjunto["estacion_id"].nunique()),
        "viajes_media": round(float(conjunto[OBJETIVO].mean()), 2),
        "viajes_mediana": float(conjunto[OBJETIVO].median()),
    }


def resumen(referencia: pd.DataFrame, actual: pd.DataFrame, puntajes: pd.Series) -> dict:
    """Periodos, estaciones nuevas y drift por variable, para el reporte de la entrega."""
    nuevas = set(actual["estacion_id"]) - set(referencia["estacion_id"])
    con_drift = puntajes[puntajes >= UMBRAL_DRIFT]
    return {
        "referencia": describir(referencia),
        "actual": describir(actual),
        "estaciones_nuevas": len(nuevas),
        "filas_de_estaciones_nuevas": int(actual["estacion_id"].isin(nuevas).sum()),
        "metodos": {"numericas": METODO_NUMERICAS, "categoricas": METODO_CATEGORICAS},
        "umbral_drift": UMBRAL_DRIFT,
        "drift_por_variable": puntajes.round(4).to_dict(),
        "variables_con_drift": list(con_drift.index),
        "proporcion_con_drift": round(len(con_drift) / len(puntajes), 3),
        "drift_del_conjunto": len(con_drift) / len(puntajes) >= PROPORCION_DRIFT,
    }


@app.command()
def p1(tabla_path: Path = P1_FEATURES_PATH, salida_dir: Path = EVIDENTLY_DIR):
    """P1: reporte de drift y calidad de datos, entrenamiento contra los 12 meses posteriores."""
    tabla = pd.read_parquet(tabla_path)
    referencia, actual = separar_periodos(tabla)
    for nombre, conjunto in (("Referencia", referencia), ("Actual", actual)):
        d = describir(conjunto)
        logger.info(f"{nombre}: {d['filas']:,} filas, {d['desde']} a {d['hasta']}")

    salida_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Report([preset_drift(), DataSummaryPreset()]).run(
        dataset(actual), dataset(referencia)
    )
    snapshot.save_html(str(salida_dir / "p1_drift_calidad.html"))
    puntajes = puntajes_drift(snapshot)
    datos_resumen = resumen(referencia, actual, puntajes)
    (salida_dir / "p1_drift_resumen.json").write_text(
        json.dumps(datos_resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(f"Drift por variable:\n{puntajes.round(3).to_string()}")
    logger.info(
        f"Variables con drift (>= {UMBRAL_DRIFT}): {datos_resumen['variables_con_drift']}; "
        f"drift del conjunto: {datos_resumen['drift_del_conjunto']}"
    )

    por_mes = drift_por_mes(referencia, actual)
    por_mes.round(4).to_csv(salida_dir / "p1_drift_por_mes.csv")
    graficar_drift_por_mes(por_mes).savefig(salida_dir / "p1_drift_por_mes.png", dpi=150)
    logger.success(f"Reportes de Evidently guardados en {salida_dir}")


if __name__ == "__main__":
    app()
