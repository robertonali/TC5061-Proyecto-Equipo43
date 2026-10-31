"""Modelos de P1 (demanda de viajes por estación y día) registrados en MLflow.

Lee la tabla de `features.py` (data/processed/p1_estacion_dia.parquet), que ya trae la partición
temporal común, entrena el modelo elegido con `--modelo` y lo evalúa en validación y prueba:

- `baseline`: el mismo día de la semana pasada (`lag_7d`), sin entrenamiento;
- `ridge`: regresión lineal con regularización, con las variables preparadas en un Pipeline;
- `rf`: RandomForestRegressor;
- `hgb`: HistGradientBoostingRegressor;
- `busqueda-hgb`: búsqueda de hiperparámetros de `hgb` en la rejilla P1_BUSQUEDA_HGB, como un run
  padre con un run hijo (nested run) por combinación. Los hijos solo se evalúan en validación.

Los hiperparámetros por defecto están en P1_HIPERPARAMETROS (config.py) y se cambian con las
opciones de la terminal (o con `-P` en `mlflow run`).

Cada run registra las fechas, las versiones de los datos, la semilla y los hiperparámetros como
parámetros; MAE y RMSE en validación y prueba (`val_*`, `test_*`), su mejora frente al baseline y
el tiempo de entrenamiento y predicción como métricas; el equipo, la pregunta, el commit, la
familia del modelo (`model_family`) y la etapa del trabajo (`stage_of_work`) como etiquetas. Como
artefactos: gráficas y tablas de error en `plots/`, el manifiesto de los datos en `data/`, la
configuración en `config/` y, en los candidatos, el modelo con su firma y un ejemplo de entrada.

Uso (normalmente con `mlflow run`, ver la sección de P1 del README):
    python -m mibici_mlops.modeling.train_p1 --modelo baseline
    python -m mibici_mlops.modeling.train_p1 --modelo hgb --max-leaf-nodes 127
    python -m mibici_mlops.modeling.train_p1 --modelo busqueda-hgb

Antes de correrlo, el servidor de MLflow debe estar levantado (ver la sección de MLflow del
README):
    mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination mlruns --port 5000 --workers 1

Los runs se ven en http://127.0.0.1:5000.
"""

import itertools
import json
from pathlib import Path
import time

from loguru import logger
from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mibici_mlops.config import (
    EXPERIMENTO_MLFLOW,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    MLFLOW_TRACKING_URI,
    P1_BUSQUEDA_HGB,
    P1_FEATURES_PATH,
    P1_HIPERPARAMETROS,
    SEMILLA,
)
from mibici_mlops.features import COLUMNAS_P1
from mibici_mlops.modeling.registro import (
    etiquetas_run,
    registrar_configuracion,
    registrar_conjuntos,
    registrar_manifiesto,
    registrar_parametros,
    ruta_para_registro,
    usar_experimento,
    version_datos,
)

app = typer.Typer()

OBJETIVO = "viajes"
CATEGORICAS = ["dia_semana", "mes"]
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

# MLflow guarda los modelos de scikit-learn con skops, que al cargarlos rechaza los tipos que no
# conoce. Los tres primeros son parte de HistGradientBoostingRegressor (igual que en P2) y el
# último, de los árboles de RandomForestRegressor.
TIPOS_CONFIABLES_SKOPS = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
    "sklearn.tree._tree.Tree",
]


def crear_ridge(alpha: float) -> Pipeline:
    """Regresión ridge: día de la semana y mes en one-hot, el resto estandarizado."""
    numericas = [c for c in COLUMNAS_P1 if c not in CATEGORICAS]
    preparacion = ColumnTransformer(
        [
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("numericas", StandardScaler(), numericas),
        ]
    )
    return Pipeline([("preparacion", preparacion), ("modelo", Ridge(alpha=alpha))])


def crear_rf(
    n_estimators: int, max_depth: int, min_samples_leaf: int, max_samples: float
) -> Pipeline:
    """Random forest. `max_samples` < 1 entrena cada árbol con una muestra para bajar el costo."""
    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        n_jobs=-1,
        random_state=SEMILLA,
    )
    return Pipeline([("modelo", modelo)])


def crear_hgb(loss: str, learning_rate: float, max_iter: int, max_leaf_nodes: int) -> Pipeline:
    """Gradient boosting. Se prueba la pérdida de Poisson porque el conteo de viajes es >= 0."""
    modelo = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        early_stopping=False,
        random_state=SEMILLA,
    )
    return Pipeline([("modelo", modelo)])


MODELOS = {
    "baseline": {"familia": "naive", "etapa": "baseline"},
    "ridge": {"familia": "linear", "etapa": "candidate", "crear": crear_ridge},
    "rf": {"familia": "random_forest", "etapa": "candidate", "crear": crear_rf},
    "hgb": {"familia": "gradient_boosting", "etapa": "candidate", "crear": crear_hgb},
}
BUSQUEDA = "busqueda-hgb"


def resolver_hiperparametros(modelo: str, opciones: dict) -> dict:
    """Hiperparámetros del modelo: los de P1_HIPERPARAMETROS, con los de la terminal encima.

    Falla si se pasa uno que el modelo no usa, para que un error de dedo no se ignore en silencio.
    En la búsqueda, la rejilla define los suyos y no se pueden pasar por la terminal.
    """
    if modelo == BUSQUEDA:
        por_defecto = {
            nombre: valor
            for nombre, valor in P1_HIPERPARAMETROS["hgb"].items()
            if nombre not in P1_BUSQUEDA_HGB
        }
    else:
        por_defecto = P1_HIPERPARAMETROS.get(modelo, {})
    pasados = {nombre: valor for nombre, valor in opciones.items() if valor is not None}
    sobrantes = set(pasados) - set(por_defecto)
    if sobrantes:
        raise typer.BadParameter(f"{modelo} no acepta: {', '.join(sorted(sobrantes))}")
    return {**por_defecto, **pasados}


def nombre_run(modelo: str, hiperparametros: dict) -> str:
    """Nombre descriptivo del run: el modelo y los hiperparámetros que lo distinguen."""
    if modelo == "baseline":
        return "p1-baseline-semana-previa"
    if modelo == "ridge":
        return f"p1-ridge-alpha{hiperparametros['alpha']}"
    if modelo == "rf":
        return (
            f"p1-rf-arboles{hiperparametros['n_estimators']}"
            f"-prof{hiperparametros['max_depth']}-hoja{hiperparametros['min_samples_leaf']}"
        )
    if modelo == BUSQUEDA:
        return (
            f"p1-busqueda-hgb-lr{hiperparametros['learning_rate']}"
            f"-arboles{hiperparametros['max_iter']}"
        )
    return (
        f"p1-hgb-{hiperparametros['loss']}-lr{hiperparametros['learning_rate']}"
        f"-hojas{hiperparametros['max_leaf_nodes']}-arboles{hiperparametros['max_iter']}"
    )


def metricas(real: pd.Series, prediccion: np.ndarray, prefijo: str) -> dict[str, float]:
    """MAE y RMSE en número de viajes por estación-día."""
    return {
        f"{prefijo}_mae": float(mean_absolute_error(real, prediccion)),
        f"{prefijo}_rmse": float(root_mean_squared_error(real, prediccion)),
    }


def predecir(modelo: Pipeline, datos: pd.DataFrame) -> np.ndarray:
    """Predicción del modelo, recortada a 0: no hay demanda negativa."""
    return np.clip(modelo.predict(datos[COLUMNAS_P1]), 0, None)


def error_por_dia_semana(
    prueba: pd.DataFrame, predicciones: dict[str, np.ndarray]
) -> pd.DataFrame:
    """MAE por día de la semana (0 = lunes) para cada modelo."""
    filas = {
        nombre: (prueba[OBJETIVO] - pred).abs().groupby(prueba["dia_semana"]).mean()
        for nombre, pred in predicciones.items()
    }
    return pd.DataFrame(filas).rename_axis("dia_semana").reset_index()


def error_por_nivel_uso(
    entrenamiento: pd.DataFrame, prueba: pd.DataFrame, predicciones: dict[str, np.ndarray]
) -> pd.DataFrame:
    """MAE por nivel de uso de la estación (terciles de la demanda media en entrenamiento).

    Los terciles se calculan solo con el entrenamiento. Las estaciones sin historial de
    entrenamiento se reportan aparte, como "sin historial".
    """
    media = entrenamiento.groupby("estacion_id")[OBJETIVO].mean()
    cortes = media.quantile([1 / 3, 2 / 3]).to_numpy()
    nivel = pd.cut(
        media, [-np.inf, cortes[0], cortes[1], np.inf], labels=["bajo", "medio", "alto"]
    )
    nivel_fila = prueba["estacion_id"].map(nivel).astype(object).fillna("sin historial")
    resultado = {"viajes_medios": prueba[OBJETIVO].groupby(nivel_fila).mean()}
    for nombre, pred in predicciones.items():
        resultado[nombre] = (prueba[OBJETIVO] - pred).abs().groupby(nivel_fila).mean()
    tabla = pd.DataFrame(resultado).rename_axis("nivel_uso").reset_index()
    tabla["filas"] = tabla["nivel_uso"].map(nivel_fila.value_counts())
    return tabla


def graficar_real_vs_predicho(
    prueba: pd.DataFrame, predicciones: dict[str, np.ndarray], titulo: str
) -> Figure:
    """Demanda real y predicha, total por día en el periodo de prueba."""
    diario = prueba[["fecha", OBJETIVO]].copy()
    for nombre, pred in predicciones.items():
        diario[nombre] = pred
    diario = diario.groupby("fecha").sum()
    figura = Figure(figsize=(10, 4))
    ax = figura.subplots()
    ax.plot(diario.index, diario[OBJETIVO], color="black", label="Real")
    for nombre in predicciones:
        ax.plot(diario.index, diario[nombre], alpha=0.8, label=nombre)
    ax.set(title=titulo, ylabel="Viajes (suma de las estaciones)")
    ax.legend()
    figura.autofmt_xdate()
    return figura


def graficar_residuales(real: pd.Series, prediccion: np.ndarray, titulo: str) -> Figure:
    """Residuales (real - predicho) contra lo predicho y su distribución."""
    residual = real.to_numpy() - prediccion
    figura = Figure(figsize=(10, 4), layout="constrained")
    ax_dispersion, ax_hist = figura.subplots(1, 2)
    ax_dispersion.scatter(prediccion, residual, s=2, alpha=0.2)
    ax_dispersion.axhline(0, color="black", linewidth=1)
    ax_dispersion.set(xlabel="Viajes predichos", ylabel="Real - predicho", title="Residuales")
    ax_hist.hist(np.clip(residual, -60, 60), bins=60)
    ax_hist.set(xlabel="Real - predicho (recortado a ±60)", ylabel="Filas", title="Distribución")
    figura.suptitle(titulo)
    return figura


def graficar_error_por_dia(por_dia: pd.DataFrame, nombres: list[str], titulo: str) -> Figure:
    figura = Figure(figsize=(8, 4))
    ax = figura.subplots()
    for nombre in nombres:
        ax.plot(por_dia["dia_semana"], por_dia[nombre], marker="o", label=nombre)
    ax.set_xticks(range(7), DIAS)
    ax.set(title=titulo, ylabel="MAE (viajes)")
    ax.legend()
    return figura


def graficar_error_por_nivel(por_nivel: pd.DataFrame, nombres: list[str], titulo: str) -> Figure:
    figura = Figure(figsize=(8, 4))
    ax = figura.subplots()
    posiciones = np.arange(len(por_nivel))
    ancho = 0.8 / len(nombres)
    for i, nombre in enumerate(nombres):
        ax.bar(posiciones + i * ancho, por_nivel[nombre], ancho, label=nombre)
    ax.set_xticks(posiciones + ancho * (len(nombres) - 1) / 2, por_nivel["nivel_uso"])
    ax.set(title=titulo, ylabel="MAE (viajes)")
    ax.legend()
    return figura


def registrar_evaluacion(
    nombre: str,
    entrenamiento: pd.DataFrame,
    conjuntos: dict[str, pd.DataFrame],
    predicciones: dict[str, np.ndarray],
) -> dict[str, float]:
    """Métricas de validación y prueba, la mejora frente al baseline y las gráficas en plots/.

    El baseline (`lag_7d`) se recalcula aquí para que cada run diga cuánto mejora sobre él.
    """
    resultado = {}
    for prefijo, datos in conjuntos.items():
        resultado.update(metricas(datos[OBJETIVO], predicciones[prefijo], prefijo))
        base = metricas(datos[OBJETIVO], datos["lag_7d"], prefijo)
        for clave, valor in base.items():
            resultado[f"{clave}_mejora_vs_baseline"] = 1 - resultado[clave] / valor

    prueba = conjuntos["test"]
    comparadas = {"baseline": prueba["lag_7d"].to_numpy()}
    if nombre != nombre_run("baseline", {}):
        comparadas[nombre] = predicciones["test"]
    nombres = list(comparadas)
    por_dia = error_por_dia_semana(prueba, comparadas)
    por_nivel = error_por_nivel_uso(entrenamiento, prueba, comparadas)
    graficas = {
        "plots/real_vs_predicho_test.png": graficar_real_vs_predicho(
            prueba, comparadas, f"P1 · {nombre} · viajes por día (prueba)"
        ),
        "plots/residuales_test.png": graficar_residuales(
            prueba[OBJETIVO], predicciones["test"], f"P1 · {nombre} (prueba)"
        ),
        "plots/error_por_dia_semana_test.png": graficar_error_por_dia(
            por_dia, nombres, f"P1 · {nombre} · MAE por día de la semana (prueba)"
        ),
        "plots/error_por_nivel_uso_test.png": graficar_error_por_nivel(
            por_nivel, nombres, f"P1 · {nombre} · MAE por nivel de uso (prueba)"
        ),
    }
    for ruta, figura in graficas.items():
        mlflow.log_figure(figura, ruta)
    mlflow.log_text(por_dia.to_csv(index=False), "plots/error_por_dia_semana_test.csv")
    mlflow.log_text(por_nivel.to_csv(index=False), "plots/error_por_nivel_uso_test.csv")
    mlflow.log_metrics(resultado)
    return resultado


def registrar_datos(
    entrenamiento: pd.DataFrame, conjuntos: dict[str, pd.DataFrame], fuente: str
) -> None:
    """Versión de los datos en data/: el manifiesto de la ingesta y los conjuntos con log_input.

    La tabla de P1 es por estación y día: no tiene datos de las personas usuarias.
    """
    registrar_manifiesto()
    columnas = COLUMNAS_P1 + [OBJETIVO]
    registrar_conjuntos(
        {
            "training": entrenamiento[columnas],
            "validation": conjuntos["val"][columnas],
            "testing": conjuntos["test"][columnas],
        },
        fuente,
        "p1_estacion_dia",
        OBJETIVO,
    )


def registrar_modelo(modelo: Pipeline, entrenamiento: pd.DataFrame) -> None:
    """Guarda el modelo entrenado con su firma (columnas y tipos de entrada y salida) y un ejemplo."""
    ejemplo = entrenamiento[COLUMNAS_P1].head(5)
    firma = infer_signature(ejemplo, modelo.predict(ejemplo))
    mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        signature=firma,
        input_example=ejemplo,
        skops_trusted_types=TIPOS_CONFIABLES_SKOPS,
    )


def correr_modelo(
    modelo: str,
    hiperparametros: dict,
    entrenamiento: pd.DataFrame,
    conjuntos: dict[str, pd.DataFrame],
    parametros: dict,
    etiquetas: dict,
) -> dict[str, float]:
    """Entrena (si hace falta), evalúa y registra un modelo como un run de MLflow."""
    elegido = MODELOS[modelo]
    nombre = nombre_run(modelo, hiperparametros)
    tags = {**etiquetas, "model_family": elegido["familia"], "stage_of_work": elegido["etapa"]}
    with mlflow.start_run(run_name=nombre, tags=tags):
        mlflow.set_tag("mlflow.runName", nombre)
        if "crear" in elegido:
            inicio = time.perf_counter()
            estimador = elegido["crear"](**hiperparametros).fit(
                entrenamiento[COLUMNAS_P1], entrenamiento[OBJETIVO]
            )
            segundos = time.perf_counter() - inicio
            inicio = time.perf_counter()
            predicciones = {p: predecir(estimador, d) for p, d in conjuntos.items()}
            tiempos = {
                "tiempo_entrenamiento_s": segundos,
                "tiempo_prediccion_s": time.perf_counter() - inicio,
            }
            hiperparametros = {
                "modelo": type(estimador[-1]).__name__,
                "variables": ",".join(COLUMNAS_P1),
                **hiperparametros,
            }
        else:
            estimador, tiempos = None, {}
            predicciones = {p: d["lag_7d"].to_numpy() for p, d in conjuntos.items()}
            hiperparametros = {"modelo": "naive_estacional_semana_previa"}
        registrar_parametros({**parametros, **hiperparametros})
        resultado = registrar_evaluacion(nombre, entrenamiento, conjuntos, predicciones)
        mlflow.log_metrics(tiempos)
        registrar_datos(entrenamiento, conjuntos, parametros["tabla"])
        registrar_configuracion(
            {"modelo": modelo, "hiperparametros": hiperparametros, "parametros": parametros}, "P1"
        )
        if estimador is not None:
            registrar_modelo(estimador, entrenamiento)
    logger.info(f"{nombre}: { {k: round(v, 3) for k, v in {**resultado, **tiempos}.items()} }")
    return resultado


def buscar_hgb(
    fijos: dict,
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    parametros: dict,
    etiquetas: dict,
) -> pd.DataFrame:
    """Búsqueda de hiperparámetros de `hgb` en la rejilla P1_BUSQUEDA_HGB, con nested runs.

    El run padre agrupa la búsqueda y guarda la mejor combinación y la tabla de resultados; cada
    combinación es un run hijo. Los hijos solo se evalúan en validación: la prueba se reserva para
    el candidato final, que se vuelve a correr con `--modelo hgb` y los mejores hiperparámetros.
    """
    combinaciones = [
        dict(zip(P1_BUSQUEDA_HGB, valores))
        for valores in itertools.product(*P1_BUSQUEDA_HGB.values())
    ]
    nombre = nombre_run(BUSQUEDA, fijos)
    tags = {**etiquetas, "model_family": "gradient_boosting", "stage_of_work": "search"}
    resultados = []
    with mlflow.start_run(run_name=nombre, tags=tags):
        mlflow.set_tag("mlflow.runName", nombre)
        registrar_parametros(
            {
                **parametros,
                **fijos,
                "modelo": "HistGradientBoostingRegressor",
                "variables": ",".join(COLUMNAS_P1),
                "rejilla": json.dumps(P1_BUSQUEDA_HGB),
                "combinaciones": len(combinaciones),
            }
        )
        for numero, combinacion in enumerate(combinaciones, start=1):
            hiperparametros = {**fijos, **combinacion}
            hijo = nombre_run("hgb", hiperparametros)
            logger.info(f"Combinación {numero}/{len(combinaciones)}: {hijo}")
            with mlflow.start_run(run_name=hijo, nested=True, tags=tags):
                mlflow.log_params(
                    {**parametros, **hiperparametros, "modelo": "HistGradientBoostingRegressor"}
                )
                inicio = time.perf_counter()
                estimador = crear_hgb(**hiperparametros).fit(
                    entrenamiento[COLUMNAS_P1], entrenamiento[OBJETIVO]
                )
                segundos = time.perf_counter() - inicio
                val = metricas(validacion[OBJETIVO], predecir(estimador, validacion), "val")
                mlflow.log_metrics({**val, "tiempo_entrenamiento_s": segundos})
            resultados.append({**combinacion, **val, "tiempo_entrenamiento_s": segundos})

        tabla = pd.DataFrame(resultados).sort_values("val_mae")
        mejor = {clave: tabla[clave].iloc[0] for clave in P1_BUSQUEDA_HGB}
        mejor = {k: v.item() if hasattr(v, "item") else v for k, v in mejor.items()}
        mlflow.log_metric("mejor_val_mae", tabla["val_mae"].iloc[0])
        mlflow.log_params({f"mejor_{clave}": valor for clave, valor in mejor.items()})
        mlflow.log_text(tabla.to_csv(index=False), "busqueda/resultados.csv")
        registrar_manifiesto()
        registrar_configuracion(
            {"modelo": BUSQUEDA, "fijos": fijos, "rejilla": P1_BUSQUEDA_HGB, "parametros": parametros},
            "P1",
        )
    logger.info(f"Mejor combinación: {mejor}")
    return tabla


@app.command()
def main(
    modelo: str = typer.Option(..., help=f"Modelo a correr: {', '.join([*MODELOS, BUSQUEDA])}."),
    alpha: float = typer.Option(None, help="ridge: regularización."),
    n_estimators: int = typer.Option(None, help="rf: número de árboles."),
    max_depth: int = typer.Option(None, help="rf: profundidad máxima."),
    min_samples_leaf: int = typer.Option(None, help="rf: filas mínimas por hoja."),
    max_samples: float = typer.Option(None, help="rf: fracción de filas por árbol."),
    loss: str = typer.Option(None, help="hgb: poisson o squared_error."),
    learning_rate: float = typer.Option(None, help="hgb: tasa de aprendizaje."),
    max_iter: int = typer.Option(None, help="hgb: número de árboles."),
    max_leaf_nodes: int = typer.Option(None, help="hgb: hojas máximas por árbol."),
    features_path: Path = P1_FEATURES_PATH,
    muestra_entrenamiento: float = typer.Option(
        1.0, help="Fracción de filas de entrenamiento a usar (solo para pruebas rápidas)."
    ),
):
    if modelo not in [*MODELOS, BUSQUEDA]:
        raise typer.BadParameter(
            f"usa uno de: {', '.join([*MODELOS, BUSQUEDA])}", param_hint="--modelo"
        )
    opciones = {
        "alpha": alpha,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_samples": max_samples,
        "loss": loss,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
    }
    hiperparametros = resolver_hiperparametros(modelo, opciones)

    tabla = pd.read_parquet(features_path)
    entrenamiento = tabla[tabla["conjunto"] == "entrenamiento"]
    conjuntos = {
        "val": tabla[tabla["conjunto"] == "validacion"],
        "test": tabla[tabla["conjunto"] == "prueba"],
    }
    if muestra_entrenamiento < 1.0:
        entrenamiento = entrenamiento.sample(frac=muestra_entrenamiento, random_state=SEMILLA)
        logger.warning(f"Entrenando con {len(entrenamiento):,} filas ({muestra_entrenamiento:.0%})")
    logger.info(
        f"Filas: {len(entrenamiento):,} de entrenamiento, {len(conjuntos['val']):,} de "
        f"validación y {len(conjuntos['test']):,} de prueba"
    )

    usar_experimento()
    etiquetas = etiquetas_run("P1", __file__)
    parametros = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "fecha_inicio_validacion": FECHA_INICIO_VALIDACION,
        "fecha_inicio_prueba": FECHA_INICIO_PRUEBA,
        "version_datos": version_datos(),
        "tabla": ruta_para_registro(features_path),
        "semilla": SEMILLA,
        "muestra_entrenamiento": muestra_entrenamiento,
        "n_filas_entrenamiento": len(entrenamiento),
        "n_filas_validacion": len(conjuntos["val"]),
        "n_filas_prueba": len(conjuntos["test"]),
    }

    if modelo == BUSQUEDA:
        buscar_hgb(hiperparametros, entrenamiento, conjuntos["val"], parametros, etiquetas)
    else:
        correr_modelo(modelo, hiperparametros, entrenamiento, conjuntos, parametros, etiquetas)
    logger.success(f"Registrado en el experimento {EXPERIMENTO_MLFLOW} ({MLFLOW_TRACKING_URI})")


if __name__ == "__main__":
    app()
