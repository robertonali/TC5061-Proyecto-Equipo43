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

P2 (riesgo de las estaciones) compara las ventanas de `entrenamiento` (referencia) con las de
`validacion` y las de `prueba` (actual): las 6 variables del modelo, la etiqueta y la clase que
predice el `champion` del Model Registry. Como P2 sí tiene etiqueta, también mide el desempeño del
modelo en cada periodo y en cada mes, junto al baseline de frecuencia. Los periodos y el porqué
están en references/decisiones_p2.md, decisión 12. Necesita el servidor de MLflow activo.

    python -m mibici_mlops.monitoreo p2
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
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
import typer

from mibici_mlops.config import (
    EVIDENTLY_DIR,
    P1_FEATURES_PATH,
    P1_MONITOREO_FIN,
    P1_MONITOREO_INICIO,
    P2_MODELO_REGISTRADO,
    P2_MONITOREO_ACTUALES,
    P2_SERVICIO_MODELO_URI,
    P2_VENTANA,
    P2_VENTANAS_PATH,
    P3_FRANJAS,
    P3_MODELO_REGISTRADO,
    P3_MONITOREO_ACTUALES,
    P3_PERFILES_PATH,
    P3_SERVICIO_MODELO_URI,
)
from mibici_mlops.features import CLASES_P2, COLUMNAS_P1, COLUMNAS_P3, DIAS_SEMANA, VARIABLES_P2
from mibici_mlops.modeling.registro import (
    etiquetas_run,
    registrar_parametros,
    usar_experimento,
    version_datos,
)
from mibici_mlops.modeling.train_p2 import (
    CLASES_RIESGO,
    COLORES,
    FONDO,
    NOMBRES,
    TEXTO,
    TEXTO_SECUNDARIO,
    asignar_conjunto,
    baseline_frecuencia,
    calcular_metricas,
    dar_estilo,
    predecir,
    probabilidades_modelo,
)

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


def graficar_drift_por_mes(por_mes: pd.DataFrame, pregunta: str = "P1") -> Figure:
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
        f"{pregunta}: drift de cada mes contra el mismo mes en entrenamiento "
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


# ---------------------------------------------------------------------------------------------
# P2: riesgo de las estaciones. Se vigilan las 6 variables que recibe el modelo, la etiqueta
# (`clase`) y la clase que predice el `champion` (drift de predicción). Como P2 sí tiene etiqueta,
# también se mide el desempeño del modelo en cada periodo, junto al baseline de frecuencia.
PREDICCION = "prediccion"
REFERENCIA_P2 = "entrenamiento"
NUMERICAS_P2 = ["capacidad", "frec_se_vacia", "frec_se_llena"]
CATEGORICAS_P2 = ["hora", "dia_semana", "feriado", "clase", PREDICCION]
COLUMNAS_MONITOREO_P2 = [*VARIABLES_P2, "clase", PREDICCION]
NOMBRES_CONJUNTO_P2 = {"entrenamiento": "entrenamiento", "validacion": "validación"}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
# Colores de la gráfica de desempeño. Las clases de riesgo usan los de train_p2 (espacios 1 y 2 de
# la paleta categórica de referencia) y el champion, el espacio 3. El baseline va en gris porque
# es el contexto, no la serie que se sigue.
COLOR_CHAMPION = "#1baf7a"
COLOR_BASELINE = "#898781"
COLOR_EJE = "#c3c2b7"


def preparar_p2(conjunto: pd.DataFrame) -> pd.DataFrame:
    """Columnas que se monitorean. Las categóricas pasan a texto, legible en el reporte."""
    datos = conjunto[["estacion", "ventana", *COLUMNAS_MONITOREO_P2]].copy()
    datos["hora"] = datos["hora"].map({hora: f"{hora:02d}:00" for hora in range(24)})
    datos["dia_semana"] = datos["dia_semana"].map(dict(enumerate(DIAS_SEMANA)))
    datos["feriado"] = np.where(datos["feriado"], "si", "no")
    return datos


def dataset_p2(conjunto: pd.DataFrame) -> Dataset:
    definicion = DataDefinition(
        numerical_columns=NUMERICAS_P2,
        categorical_columns=CATEGORICAS_P2,
        datetime_columns=["ventana"],
        id_column="estacion",
    )
    return Dataset.from_pandas(preparar_p2(conjunto), data_definition=definicion)


def drift_por_mes_p2(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Drift de cada mes de `actual` contra el mismo mes del año en `referencia`.

    Como en P1: comparar diciembre de 2025 con los diciembres de entrenamiento quita la
    estacionalidad, y lo que queda es el cambio de un año a otro. Regresa una tabla variable × mes
    con la distancia de drift.
    """
    resultados = {}
    for periodo, mes_actual in actual.groupby(actual["dia"].dt.to_period("M")):
        mismo_mes = referencia[referencia["dia"].dt.month == periodo.month]
        snapshot = Report([preset_drift(COLUMNAS_MONITOREO_P2)]).run(
            dataset_p2(mes_actual), dataset_p2(mismo_mes)
        )
        resultados[str(periodo)] = puntajes_drift(snapshot)
    return pd.DataFrame(resultados).loc[COLUMNAS_MONITOREO_P2]


def proporcion_clases(clases: pd.Series) -> dict[str, float]:
    """Fracción de las ventanas en cada clase, en el orden de CLASES_P2."""
    proporciones = clases.value_counts(normalize=True).reindex(CLASES_P2, fill_value=0.0)
    return proporciones.round(4).to_dict()


def calidad_p2(conjunto: pd.DataFrame) -> dict[str, int]:
    """Chequeos de calidad propios de la tabla de P2; complementan los de DataSummaryPreset.

    Todos deben dar 0: sin valores faltantes, una sola fila por estación y ventana, frecuencias
    entre 0 y 1, horas en la rejilla de P2_VENTANA, capacidad positiva y solo las 3 clases.
    """
    frecuencias = conjunto[["frec_se_vacia", "frec_se_llena"]]
    horas_por_ventana = pd.Timedelta(P2_VENTANA) // pd.Timedelta(hours=1)
    return {
        "valores_faltantes": int(conjunto[[*VARIABLES_P2, "clase"]].isna().sum().sum()),
        "ventanas_duplicadas": int(conjunto.duplicated(["estacion", "ventana"]).sum()),
        "frecuencias_fuera_de_0_a_1": int(
            ((frecuencias < 0) | (frecuencias > 1)).any(axis=1).sum()
        ),
        "horas_fuera_de_la_rejilla": int((conjunto["hora"] % horas_por_ventana != 0).sum()),
        "capacidad_no_positiva": int((conjunto["capacidad"] <= 0).sum()),
        "clases_desconocidas": int((~conjunto["clase"].isin(CLASES_P2)).sum()),
    }


def describir_p2(conjunto: pd.DataFrame) -> dict:
    """Periodo, tamaño, clases reales y predichas, y calidad de un conjunto de ventanas."""
    return {
        "desde": f"{conjunto['dia'].min():%Y-%m-%d}",
        "hasta": f"{conjunto['dia'].max():%Y-%m-%d}",
        "ventanas": len(conjunto),
        "estaciones": int(conjunto["estacion"].nunique()),
        "proporcion_clases": proporcion_clases(conjunto["clase"]),
        "proporcion_predicha": proporcion_clases(conjunto[PREDICCION]),
        "calidad": calidad_p2(conjunto),
    }


def desempeno_p2(conjunto: pd.DataFrame, probabilidades: pd.DataFrame) -> dict[str, float] | None:
    """Métricas del modelo en `conjunto` y PR-AUC promedio del baseline de frecuencia.

    Son las métricas del entrenamiento (train_p2.calcular_metricas). El baseline da la escala: con
    menos ventanas en riesgo la PR-AUC baja para cualquier modelo, así que lo que dice si el
    `champion` se degrada es su ventaja sobre el baseline (`mejora_sobre_baseline`). Regresa None
    si al conjunto le falta alguna clase de riesgo, porque no se pueden calcular.
    """
    if not set(CLASES_RIESGO) <= set(conjunto["clase"].unique()):
        return None
    modelo = calcular_metricas(conjunto["clase"], probabilidades.loc[conjunto.index], "modelo")
    baseline = calcular_metricas(
        conjunto["clase"], baseline_frecuencia(conjunto, conjunto), "baseline"
    )
    metricas = {clave.removeprefix("modelo_"): float(valor) for clave, valor in modelo.items()}
    metricas["pr_auc_promedio_baseline"] = float(baseline["baseline_pr_auc_promedio"])
    metricas["mejora_sobre_baseline"] = (
        metricas["pr_auc_promedio"] - metricas["pr_auc_promedio_baseline"]
    )
    return {clave: round(valor, 4) for clave, valor in metricas.items()}


def desempeno_por_mes_p2(datos: pd.DataFrame, probabilidades: pd.DataFrame) -> pd.DataFrame:
    """`desempeno_p2` de cada mes de `datos`: una fila por mes, con su conjunto y sus ventanas."""
    filas = [
        {
            "mes": str(mes),
            "conjunto": grupo["conjunto"].iloc[0],
            "ventanas": len(grupo),
            **(desempeno_p2(grupo, probabilidades) or {}),
        }
        for mes, grupo in datos.groupby(datos["dia"].dt.to_period("M"))
    ]
    return pd.DataFrame(filas)


def resumen_p2(referencia: pd.DataFrame, actual: pd.DataFrame, puntajes: pd.Series) -> dict:
    """Estaciones que entran y salen, y drift por variable, de un periodo actual."""
    con_drift = puntajes[puntajes >= UMBRAL_DRIFT]
    ids_ref, ids_act = set(referencia["estacion"]), set(actual["estacion"])
    return {
        "estaciones_nuevas": len(ids_act - ids_ref),
        "ventanas_de_estaciones_nuevas": int(actual["estacion"].isin(ids_act - ids_ref).sum()),
        "estaciones_que_salen": len(ids_ref - ids_act),
        "drift_por_variable": puntajes.round(4).to_dict(),
        "variables_con_drift": list(con_drift.index),
        "proporcion_con_drift": round(len(con_drift) / len(puntajes), 3),
        "drift_del_conjunto": bool(len(con_drift) / len(puntajes) >= PROPORCION_DRIFT),
        "drift_de_la_prediccion": round(float(puntajes[PREDICCION]), 4),
    }


def graficar_desempeno_p2(por_mes: pd.DataFrame, version: str) -> Figure:
    """PR-AUC promedio del champion y del baseline, y ventanas en riesgo, por mes.

    Dos paneles con el mismo eje de meses, en lugar de un eje doble: la PR-AUC y el porcentaje de
    ventanas en riesgo tienen escalas distintas, y alinear dos ejes inventaría una relación.
    """
    figura = Figure(figsize=(10, 6.4), facecolor=FONDO)
    arriba, abajo = figura.subplots(2, 1, sharex=True, height_ratios=[3, 2])
    x = np.arange(len(por_mes))
    paneles = [
        (
            arriba,
            "PR-AUC promedio de «se vacía» y «se llena»",
            "{:.3f}",
            [
                (por_mes["pr_auc_promedio"], f"Champion (versión {version})", COLOR_CHAMPION),
                (por_mes["pr_auc_promedio_baseline"], "Baseline de frecuencia", COLOR_BASELINE),
            ],
        ),
        (
            abajo,
            "Ventanas en riesgo (% de las ventanas del mes)",
            "{:.1f}%",
            [
                (por_mes[f"proporcion_{riesgo}"] * 100, NOMBRES[riesgo], COLORES[riesgo])
                for riesgo in CLASES_RIESGO
            ],
        ),
    ]
    for eje, titulo, formato, series in paneles:
        for valores, nombre, color in series:
            eje.plot(
                x,
                valores,
                color=color,
                linewidth=2,
                marker="o",
                markersize=8,
                markeredgecolor=FONDO,
                markeredgewidth=1.5,
                label=nombre,
            )
        # Valor del último mes al final de cada línea, si no se enciman; los nombres van en la
        # leyenda.
        ultimos = [valores.iloc[-1] for valores, _, _ in series]
        minimo = min(valores.min() for valores, _, _ in series)
        maximo = max(valores.max() for valores, _, _ in series)
        if abs(ultimos[0] - ultimos[1]) > 0.08 * (maximo - minimo):
            for ultimo in ultimos:
                eje.annotate(
                    formato.format(ultimo),
                    (x[-1], ultimo),
                    xytext=(10, 0),
                    textcoords="offset points",
                    va="center",
                    color=TEXTO_SECUNDARIO,
                    fontsize=9,
                )
        eje.set_title(titulo, loc="left", color=TEXTO, fontsize=10)
        eje.legend(loc="lower left", ncols=2, frameon=False, labelcolor=TEXTO_SECUNDARIO)
        dar_estilo(eje)
    valores = por_mes[["pr_auc_promedio", "pr_auc_promedio_baseline"]].to_numpy()
    arriba.set_ylim(np.floor(valores.min() * 20 - 1) / 20, np.ceil(valores.max() * 20 + 0.5) / 20)
    riesgo_maximo = por_mes[[f"proporcion_{riesgo}" for riesgo in CLASES_RIESGO]].to_numpy().max()
    abajo.set_ylim(0, np.ceil(riesgo_maximo * 100 + 1))
    abajo.yaxis.set_major_formatter(lambda valor, _: f"{valor:.0f}%")
    meses = pd.PeriodIndex(por_mes["mes"], freq="M")
    abajo.set_xticks(x, [f"{MESES[mes.month - 1]} {mes.year % 100}" for mes in meses])
    abajo.set_xlim(-0.5, len(x) - 1 + 1.1)

    # Línea entre los dos periodos actuales (validación y prueba), si la tabla trae los dos.
    conjuntos = por_mes["conjunto"].to_numpy()
    cambios = np.flatnonzero(conjuntos[1:] != conjuntos[:-1])
    if len(cambios):
        corte = cambios[0] + 0.5
        for eje in (arriba, abajo):
            eje.axvline(corte, color=COLOR_EJE, linewidth=1)
        nombres = [NOMBRES_CONJUNTO_P2.get(c, c) for c in (conjuntos[0], conjuntos[-1])]
        for texto, alineacion in ((f"{nombres[0]}  ", "right"), (f"  {nombres[1]}", "left")):
            arriba.text(
                corte,
                0.97,
                texto,
                transform=arriba.get_xaxis_transform(),
                ha=alineacion,
                va="top",
                color=TEXTO_SECUNDARIO,
                fontsize=9,
            )
    figura.suptitle(
        "P2 · desempeño del champion y riesgo real, por mes", x=0.01, ha="left", color=TEXTO
    )
    figura.tight_layout()
    return figura


@app.command()
def p2(ventanas_path: Path = P2_VENTANAS_PATH, salida_dir: Path = EVIDENTLY_DIR):
    """P2: drift y calidad de las ventanas y desempeño del champion; queda como run de MLflow."""
    tabla = pd.read_parquet(ventanas_path)
    tabla["conjunto"] = asignar_conjunto(tabla["dia"])
    usar_experimento()
    version = MlflowClient().get_model_version_by_alias(P2_MODELO_REGISTRADO, "champion").version
    modelo = mlflow.sklearn.load_model(P2_SERVICIO_MODELO_URI)
    logger.info(f"Modelo: {P2_SERVICIO_MODELO_URI} (versión {version})")
    probabilidades = probabilidades_modelo(modelo, tabla)
    tabla[PREDICCION] = predecir(probabilidades)
    conjuntos = {
        nombre: tabla[tabla["conjunto"] == nombre]
        for nombre in (REFERENCIA_P2, *P2_MONITOREO_ACTUALES)
    }
    for nombre, conjunto in conjuntos.items():
        logger.info(
            f"{nombre}: {len(conjunto):,} ventanas de {conjunto['estacion'].nunique()} estaciones"
        )

    referencia = conjuntos[REFERENCIA_P2]
    datos_referencia = dataset_p2(referencia)
    estaciones_referencia = set(referencia["estacion"])
    salida_dir.mkdir(parents=True, exist_ok=True)
    archivos, periodos, puntajes_por_conjunto = [], {}, {}
    for nombre in P2_MONITOREO_ACTUALES:
        actual = conjuntos[nombre]
        snapshot = Report([preset_drift(COLUMNAS_MONITOREO_P2), DataSummaryPreset()]).run(
            dataset_p2(actual), datos_referencia
        )
        html = salida_dir / f"p2_drift_calidad_{nombre}.html"
        snapshot.save_html(str(html))
        archivos.append(html)
        puntajes = puntajes_drift(snapshot)
        puntajes_por_conjunto[nombre] = puntajes
        conocidas = actual["estacion"].isin(estaciones_referencia)
        periodos[nombre] = {
            **describir_p2(actual),
            **resumen_p2(referencia, actual, puntajes),
            "desempeno": desempeno_p2(actual, probabilidades),
            "desempeno_estaciones_conocidas": desempeno_p2(actual[conocidas], probabilidades),
            "desempeno_estaciones_nuevas": desempeno_p2(actual[~conocidas], probabilidades),
        }
        logger.info(
            f"{nombre}: {len(periodos[nombre]['variables_con_drift'])} de {len(puntajes)} "
            f"variables con drift (>= {UMBRAL_DRIFT}): {periodos[nombre]['variables_con_drift']}; "
            f"PR-AUC promedio {periodos[nombre]['desempeno']['pr_auc_promedio']} "
            f"(baseline {periodos[nombre]['desempeno']['pr_auc_promedio_baseline']})"
        )

    datos_resumen = {
        "modelo": {"uri": P2_SERVICIO_MODELO_URI, "version": version},
        "metodos": {"numericas": METODO_NUMERICAS, "categoricas": METODO_CATEGORICAS},
        "umbral_drift": UMBRAL_DRIFT,
        REFERENCIA_P2: {
            **describir_p2(referencia),
            # El modelo se ajustó con estas ventanas: su desempeño aquí es optimista.
            "desempeno_dentro_de_muestra": desempeno_p2(referencia, probabilidades),
        },
        **periodos,
    }
    resumen_json = salida_dir / "p2_drift_resumen.json"
    resumen_json.write_text(
        json.dumps(datos_resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    variables_csv = salida_dir / "p2_drift_variables.csv"
    pd.DataFrame(puntajes_por_conjunto).loc[COLUMNAS_MONITOREO_P2].round(4).to_csv(variables_csv)

    # Por mes: drift contra el mismo mes de entrenamiento y desempeño del champion.
    actual_completo = pd.concat([conjuntos[nombre] for nombre in P2_MONITOREO_ACTUALES])
    drift_mensual = drift_por_mes_p2(referencia, actual_completo)
    drift_csv = salida_dir / "p2_drift_por_mes.csv"
    drift_mensual.round(4).to_csv(drift_csv)
    drift_png = salida_dir / "p2_drift_por_mes.png"
    graficar_drift_por_mes(drift_mensual, "P2").savefig(drift_png, dpi=150)
    desempeno_mensual = desempeno_por_mes_p2(actual_completo, probabilidades)
    desempeno_csv = salida_dir / "p2_desempeno_por_mes.csv"
    desempeno_mensual.to_csv(desempeno_csv, index=False)
    desempeno_png = salida_dir / "p2_desempeno_por_mes.png"
    graficar_desempeno_p2(desempeno_mensual, version).savefig(desempeno_png, dpi=150)
    archivos += [resumen_json, variables_csv, drift_csv, drift_png, desempeno_csv, desempeno_png]

    with mlflow.start_run(run_name="p2-monitoreo"):
        mlflow.set_tags(
            {
                **etiquetas_run("P2", __file__),
                "stage_of_work": "monitoring",
                "mlflow.runName": "p2-monitoreo",
            }
        )
        registrar_parametros(
            {
                "modelo_uri": P2_SERVICIO_MODELO_URI,
                "modelo_version": version,
                "referencia": REFERENCIA_P2,
                "actuales": ",".join(P2_MONITOREO_ACTUALES),
                "umbral_drift": UMBRAL_DRIFT,
                "proporcion_drift": PROPORCION_DRIFT,
                "metodo_numericas": METODO_NUMERICAS,
                "metodo_categoricas": METODO_CATEGORICAS,
                "version_datos": version_datos(),
            }
        )
        metricas = {}
        for nombre, datos in periodos.items():
            metricas[f"{nombre}_variables_con_drift"] = len(datos["variables_con_drift"])
            metricas[f"{nombre}_drift_del_conjunto"] = float(datos["drift_del_conjunto"])
            metricas[f"{nombre}_estaciones_nuevas"] = datos["estaciones_nuevas"]
            for variable, distancia in datos["drift_por_variable"].items():
                metricas[f"{nombre}_drift_{variable}"] = distancia
            for metrica, valor in datos["desempeno"].items():
                metricas[f"{nombre}_{metrica}"] = valor
        mlflow.log_metrics(metricas)
        for archivo in archivos:
            mlflow.log_artifact(str(archivo), artifact_path="evidently")
    logger.success(f"Reportes de Evidently guardados en {salida_dir} y registrados en MLflow")


if __name__ == "__main__":
    app()
