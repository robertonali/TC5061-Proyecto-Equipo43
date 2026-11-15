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

P3 (grupos de estaciones) compara los perfiles de uso de `entrenamiento` (referencia) con los de
`validacion` y `prueba` (actual), y los grupos que asigna el `champion` del Model Registry. Los
periodos y el porqué están en references/decisiones_p3.md, decisión 8. Necesita el servidor de
MLflow activo: lee el modelo del registro y deja el monitoreo como un run del experimento.

    python -m mibici_mlops.monitoreo p3
"""

import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
import typer

from mibici_mlops.config import (
    EVIDENTLY_DIR,
    P1_FEATURES_PATH,
    P1_MONITOREO_FIN,
    P1_MONITOREO_INICIO,
    P3_FRANJAS,
    P3_MODELO_REGISTRADO,
    P3_MONITOREO_ACTUALES,
    P3_PERFILES_PATH,
    P3_SERVICIO_MODELO_URI,
)
from mibici_mlops.features import COLUMNAS_P1, COLUMNAS_P3, DIAS_SEMANA
from mibici_mlops.modeling.registro import etiquetas_run, registrar_parametros, usar_experimento

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


# ---------------------------------------------------------------------------------------------
# P3: grupos de estaciones. No hay variable objetivo, así que se vigilan las 56 variables del
# perfil de uso (las que recibe el modelo) y el grupo que el modelo asigna (drift de predicción).
GRUPO = "grupo"
REFERENCIA_P3 = "entrenamiento"


def asignar_grupo(perfiles: pd.DataFrame, modelo) -> pd.DataFrame:
    """Copia de `perfiles` con el grupo que `modelo` asigna a cada estación, como texto."""
    conjunto = perfiles.copy()
    conjunto[GRUPO] = modelo.predict(conjunto[COLUMNAS_P3].astype("float64")).astype(str)
    return conjunto


def dataset_p3(conjunto: pd.DataFrame) -> Dataset:
    definicion = DataDefinition(
        numerical_columns=COLUMNAS_P3, categorical_columns=[GRUPO], id_column="estacion"
    )
    datos = conjunto[["estacion", *COLUMNAS_P3, GRUPO]].copy()
    datos[COLUMNAS_P3] = datos[COLUMNAS_P3].astype("float64")
    return Dataset.from_pandas(datos, data_definition=definicion)


def matriz_drift(puntajes: pd.Series, prefijo: str) -> pd.DataFrame:
    """Distancia de drift del perfil `prefijo` (sal o lle) como tabla día de la semana × franja."""
    tabla = pd.DataFrame(
        {
            franja: [puntajes[f"{prefijo}_{dia}_{franja}"] for dia in DIAS_SEMANA]
            for franja in P3_FRANJAS
        },
        index=DIAS_SEMANA,
    )
    return tabla


def salud_del_modelo(conjunto: pd.DataFrame) -> dict:
    """Silhouette y Davies-Bouldin de los grupos asignados, en el espacio del perfil de uso.
    Sin etiquetas reales, es la señal de si los grupos siguen separando bien a las estaciones.
    Con un solo grupo no se pueden calcular y regresa vacío.
    """
    etiquetas = conjunto[GRUPO]
    if not 1 < etiquetas.nunique() < len(conjunto):
        return {}
    perfil = conjunto[COLUMNAS_P3].astype("float64")
    return {
        "silhouette": round(float(silhouette_score(perfil, etiquetas)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(perfil, etiquetas)), 4),
    }


def proporcion_grupos(conjunto: pd.DataFrame) -> pd.Series:
    """Fracción de las estaciones de `conjunto` en cada grupo."""
    return conjunto[GRUPO].value_counts(normalize=True).sort_index()


def resumen_p3(referencia: pd.DataFrame, actual: pd.DataFrame, puntajes: pd.Series) -> dict:
    """Estaciones, drift por variable, grupos y salud del modelo de un periodo actual."""
    con_drift = puntajes[puntajes >= UMBRAL_DRIFT]
    ids_ref, ids_act = set(referencia["estacion"]), set(actual["estacion"])
    return {
        "estaciones": len(actual),
        "estaciones_nuevas": sorted(int(i) for i in ids_act - ids_ref),
        "estaciones_que_salen": sorted(int(i) for i in ids_ref - ids_act),
        "drift_por_variable": puntajes.round(4).to_dict(),
        "variables_con_drift": list(con_drift.index),
        "proporcion_con_drift": round(len(con_drift) / len(puntajes), 3),
        "drift_del_conjunto": len(con_drift) / len(puntajes) >= PROPORCION_DRIFT,
        "drift_del_grupo": float(puntajes[GRUPO]),
        "proporcion_grupos": proporcion_grupos(actual).round(4).to_dict(),
        "salud_del_modelo": salud_del_modelo(actual),
    }


def graficar_drift_franjas(por_conjunto: dict[str, pd.Series]) -> Figure:
    """Mapas de calor día × franja del drift de salidas y de llegadas, uno por periodo actual."""
    figura = Figure(figsize=(9, 3.2 * len(por_conjunto)), layout="constrained")
    ejes = figura.subplots(len(por_conjunto), 2, squeeze=False)
    maximo = max(0.3, *(float(p[COLUMNAS_P3].max()) for p in por_conjunto.values()))
    imagen = None
    for fila, (nombre, puntajes) in enumerate(por_conjunto.items()):
        for columna, (prefijo, titulo) in enumerate((("sal", "Salidas"), ("lle", "Llegadas"))):
            eje = ejes[fila][columna]
            tabla = matriz_drift(puntajes, prefijo)
            imagen = eje.imshow(tabla.to_numpy(), cmap=AZULES, vmin=0, vmax=maximo, aspect="auto")
            eje.set_xticks(range(tabla.shape[1]), tabla.columns)
            eje.set_yticks(range(tabla.shape[0]), tabla.index)
            for i, dia in enumerate(tabla.index):
                for j, franja in enumerate(tabla.columns):
                    valor = tabla.loc[dia, franja]
                    if valor >= UMBRAL_DRIFT:
                        color = "white" if valor > maximo * 0.55 else "#1f1f1f"
                        eje.text(
                            j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8, color=color
                        )
            eje.tick_params(length=0)
            for borde in eje.spines.values():
                borde.set_visible(False)
            eje.set_title(f"{titulo}: {nombre} contra {REFERENCIA_P3}", loc="left", fontsize=10)
    barra = figura.colorbar(imagen, ax=ejes, shrink=0.6)
    barra.set_label("Distancia de drift (Wasserstein normalizada)")
    barra.ax.axhline(UMBRAL_DRIFT, color="#1f1f1f", linewidth=1)
    return figura


def graficar_grupos(proporciones: pd.DataFrame) -> Figure:
    """Barras apiladas con la fracción de estaciones en cada grupo, por conjunto."""
    figura = Figure(figsize=(7, 3.6), layout="constrained")
    eje = figura.subplots()
    base = pd.Series(0.0, index=proporciones.index)
    azules = [AZULES(x) for x in (0.35, 0.65, 0.95, 0.5, 0.8, 0.2, 0.9, 0.55)]
    for i, grupo in enumerate(proporciones.columns):
        eje.barh(
            proporciones.index,
            proporciones[grupo],
            left=base,
            color=azules[i % 8],
            edgecolor="white",
            label=f"Grupo {grupo}",
        )
        base = base + proporciones[grupo]
    eje.invert_yaxis()
    eje.set_xlim(0, 1)
    eje.set_xlabel("Fracción de estaciones")
    eje.legend(
        ncols=len(proporciones.columns),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    for borde in eje.spines.values():
        borde.set_visible(False)
    return figura


def periodo_p3(perfiles: pd.DataFrame, nombre: str) -> str:
    """Texto con el periodo y las estaciones de un conjunto de perfiles, para el registro."""
    return f"{nombre}: {int((perfiles['conjunto'].astype(str) == nombre).sum())} estaciones"


@app.command()
def p3(perfiles_path: Path = P3_PERFILES_PATH, salida_dir: Path = EVIDENTLY_DIR):
    """P3: drift y calidad de los perfiles y de los grupos del champion, y registro en MLflow."""
    perfiles = pd.read_parquet(perfiles_path)
    perfiles["conjunto"] = perfiles["conjunto"].astype(str)
    conjuntos = {
        n: perfiles[perfiles["conjunto"] == n] for n in (REFERENCIA_P3, *P3_MONITOREO_ACTUALES)
    }
    for nombre, conjunto in conjuntos.items():
        logger.info(f"{nombre}: {len(conjunto)} estaciones")
    usar_experimento()
    version = MlflowClient().get_model_version_by_alias(P3_MODELO_REGISTRADO, "champion").version
    modelo = mlflow.sklearn.load_model(P3_SERVICIO_MODELO_URI)
    logger.info(f"Modelo: {P3_SERVICIO_MODELO_URI} (versión {version})")
    conjuntos = {nombre: asignar_grupo(c, modelo) for nombre, c in conjuntos.items()}
    referencia = conjuntos[REFERENCIA_P3]
    salida_dir.mkdir(parents=True, exist_ok=True)
    archivos, datos_resumen, puntajes_por_conjunto = [], {}, {}
    for nombre in P3_MONITOREO_ACTUALES:
        actual = conjuntos[nombre]
        snapshot = Report([preset_drift([*COLUMNAS_P3, GRUPO]), DataSummaryPreset()]).run(
            dataset_p3(actual), dataset_p3(referencia)
        )
        html = salida_dir / f"p3_drift_calidad_{nombre}.html"
        snapshot.save_html(str(html))
        archivos.append(html)
        puntajes = puntajes_drift(snapshot)
        puntajes_por_conjunto[nombre] = puntajes
        datos_resumen[nombre] = resumen_p3(referencia, actual, puntajes)
        logger.info(
            f"{nombre}: {len(datos_resumen[nombre]['variables_con_drift'])} de {len(puntajes)} "
            f"variables con drift (>= {UMBRAL_DRIFT}); drift del grupo: "
            f"{datos_resumen[nombre]['drift_del_grupo']:.3f}"
        )
    datos_resumen = {
        "modelo": {"uri": P3_SERVICIO_MODELO_URI, "version": version},
        "metodos": {"numericas": METODO_NUMERICAS, "categoricas": METODO_CATEGORICAS},
        "umbral_drift": UMBRAL_DRIFT,
        REFERENCIA_P3: {
            "estaciones": len(referencia),
            "proporcion_grupos": proporcion_grupos(referencia).round(4).to_dict(),
            "salud_del_modelo": salud_del_modelo(referencia),
        },
        **datos_resumen,
    }
    resumen_json = salida_dir / "p3_drift_resumen.json"
    resumen_json.write_text(
        json.dumps(datos_resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    variables_csv = salida_dir / "p3_drift_variables.csv"
    pd.DataFrame(puntajes_por_conjunto).round(4).to_csv(variables_csv)
    franjas_png = salida_dir / "p3_drift_franjas.png"
    graficar_drift_franjas(puntajes_por_conjunto).savefig(franjas_png, dpi=150)
    grupos_png = salida_dir / "p3_grupos.png"
    graficar_grupos(
        pd.DataFrame({n: proporcion_grupos(c) for n, c in conjuntos.items()}).T.fillna(0)
    ).savefig(grupos_png, dpi=150)
    archivos += [resumen_json, variables_csv, franjas_png, grupos_png]
    with mlflow.start_run(run_name="p3-monitoreo"):
        mlflow.set_tags(
            {
                **etiquetas_run("P3", __file__),
                "stage_of_work": "monitoring",
                "mlflow.runName": "p3-monitoreo",
            }
        )
        registrar_parametros(
            {
                "modelo_uri": P3_SERVICIO_MODELO_URI,
                "modelo_version": version,
                "referencia": REFERENCIA_P3,
                "actuales": ",".join(P3_MONITOREO_ACTUALES),
                "umbral_drift": UMBRAL_DRIFT,
                "proporcion_drift": PROPORCION_DRIFT,
                "metodo_numericas": METODO_NUMERICAS,
                "metodo_categoricas": METODO_CATEGORICAS,
            }
        )
        for nombre, datos in (
            (REFERENCIA_P3, datos_resumen[REFERENCIA_P3]),
            *((n, datos_resumen[n]) for n in P3_MONITOREO_ACTUALES),
        ):
            for metrica, valor in datos["salud_del_modelo"].items():
                mlflow.log_metric(f"{nombre}_{metrica}", valor)
            if nombre == REFERENCIA_P3:
                continue
            mlflow.log_metrics(
                {
                    f"{nombre}_proporcion_con_drift": datos["proporcion_con_drift"],
                    f"{nombre}_variables_con_drift": len(datos["variables_con_drift"]),
                    f"{nombre}_drift_del_grupo": datos["drift_del_grupo"],
                    f"{nombre}_drift_del_conjunto": float(datos["drift_del_conjunto"]),
                    f"{nombre}_estaciones_nuevas": len(datos["estaciones_nuevas"]),
                }
            )
        for archivo in archivos:
            mlflow.log_artifact(str(archivo), artifact_path="evidently")
    logger.success(f"Reportes de Evidently guardados en {salida_dir} y registrados en MLflow")


if __name__ == "__main__":
    app()
