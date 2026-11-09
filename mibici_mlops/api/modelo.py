"""Recupera del Model Registry los modelos que sirve la API: P1 (demanda), P2 (riesgo) y P3
(grupos de estaciones).

Cada modelo se identifica por nombre y alias (el `champion` de su pregunta: SERVICIO_MODELO_URI y
P2_SERVICIO_MODELO_URI y P3_SERVICIO_MODELO_URI en config.py) o por nombre y número de versión
(models:/mibici-equipo43-p1-demanda/3). Hay dos formas de llegar a él, y la API usa la primera
que aplique:

1. Si la variable de entorno del modelo está definida (MODELO_URI para P1, MODELO_URI_P2 para P2,
   MODELO_URI_P3 para P3),
   se carga del registry en el momento de arrancar (p. ej.
   MODELO_URI=models:/mibici-equipo43-p1-demanda@champion con
   MLFLOW_TRACKING_URI=http://host.containers.internal:5000). Requiere el servidor de MLflow.
2. Si no, se carga la copia que `exportar` descargó del registry en models/servicio/<pregunta>/ y
   que se empaqueta en la imagen. Es la opción elegida (ver la sección de despliegue del README):
   el contenedor no depende de que el servidor de MLflow esté corriendo.

Junto a cada modelo, `exportar` guarda version.json con el nombre, alias, versión y run de origen,
para que la API diga qué versión del registry está sirviendo.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.api.modelo exportar                  # P1
    python -m mibici_mlops.api.modelo exportar --question p2    # P2
    python -m mibici_mlops.api.modelo exportar --question p3    # P3
    python -m mibici_mlops.api.modelo exportar --uri models:/mibici-equipo43-p1-demanda@challenger
    python -m mibici_mlops.api.modelo exportar --question p2 --uri models:/mibici-equipo43-p2-riesgo/2
    python -m mibici_mlops.api.modelo probar
    python -m mibici_mlops.api.modelo probar --question p2
    python -m mibici_mlops.api.modelo probar --question p3
"""

import json
import os
from pathlib import Path
import shutil
from typing import Annotated

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from mibici_mlops.config import (
    P1_FEATURES_PATH,
    P2_SERVICIO_MODELO_DIR,
    P2_SERVICIO_MODELO_URI,
    P2_VENTANAS_PATH,
    P3_PERFILES_PATH,
    P3_SERVICIO_MODELO_DIR,
    P3_SERVICIO_MODELO_URI,
    SERVICIO_MODELO_DIR,
    SERVICIO_MODELO_URI,
)
from mibici_mlops.features import COLUMNAS_P1, COLUMNAS_P3, VARIABLES_P2

app = typer.Typer()


FORMATOS_URI = "models:/<nombre>@<alias> o models:/<nombre>/<versión>"

# Qué sirve la API por cada pregunta: el modelo del registry, la carpeta donde lo deja `exportar`,
# la variable de entorno que lo carga del registry en vivo, cómo se carga y con qué datos se prueba.
SERVICIOS = {
    "p1": {
        "uri": SERVICIO_MODELO_URI,
        "directorio": SERVICIO_MODELO_DIR,
        "variable": "MODELO_URI",
        # pyfunc basta: la API de P1 solo usa `predict`.
        "cargador": mlflow.pyfunc.load_model,
        "datos": P1_FEATURES_PATH,
        "columnas": COLUMNAS_P1,
        "objetivo": "viajes",
    },
    "p2": {
        "uri": P2_SERVICIO_MODELO_URI,
        "directorio": P2_SERVICIO_MODELO_DIR,
        "variable": "MODELO_URI_P2",
        # El estimador de scikit-learn, porque la API de P2 también devuelve `predict_proba`.
        "cargador": mlflow.sklearn.load_model,
        "datos": P2_VENTANAS_PATH,
        "columnas": VARIABLES_P2,
        "objetivo": "clase",
    },
    "p3": {
        "uri": P3_SERVICIO_MODELO_URI,
        "directorio": P3_SERVICIO_MODELO_DIR,
        "variable": "MODELO_URI_P3",
        # pyfunc basta: la API de P3 solo usa `predict` (el número de grupo).
        "cargador": mlflow.pyfunc.load_model,
        "datos": P3_PERFILES_PATH,
        "columnas": COLUMNAS_P3,
        # Sin variable objetivo: los grupos no tienen etiqueta.
        "objetivo": None,
    },
}


def servicio_de(question: str) -> dict:
    """La configuración del servicio de una pregunta; falla con las opciones si no existe."""
    if question not in SERVICIOS:
        raise ValueError(f"La API no sirve {question!r}; opciones: {', '.join(SERVICIOS)}")
    return SERVICIOS[question]


def separar_uri(uri: str) -> tuple[str, str | None, str | None]:
    """Nombre, alias y versión de un URI del registry; uno de alias o versión es None.

    models:/<nombre>@<alias> identifica el modelo por su alias (p. ej. champion) y
    models:/<nombre>/<versión> por su número de versión (p. ej. 3).
    """
    resto = uri.removeprefix("models:/")
    if resto == uri or not resto:
        raise ValueError(f"Se esperaba {FORMATOS_URI}, no {uri}")
    if "@" in resto:
        nombre, alias = resto.split("@", 1)
        if nombre and alias and "/" not in resto:
            return nombre, alias, None
    elif resto.count("/") == 1:
        nombre, version = resto.split("/")
        if nombre and version.isdigit():
            return nombre, None, version
    raise ValueError(f"Se esperaba {FORMATOS_URI}, no {uri}")


def metadatos(uri: str) -> dict:
    """Nombre, alias, versión y run de origen del modelo al que apunta el URI en el registry."""
    nombre, alias, version = separar_uri(uri)
    cliente = MlflowClient()
    if alias is not None:
        modelo = cliente.get_model_version_by_alias(nombre, alias)
    else:
        modelo = cliente.get_model_version(nombre, version)
    return {
        "nombre": nombre,
        "alias": alias,
        "version": modelo.version,
        "run_id": modelo.run_id,
    }


def cargar_modelo(question: str = "p1") -> tuple[object, dict]:
    """El modelo que sirve la API para una pregunta y sus metadatos (nombre, alias, versión, run).

    Se carga con el `cargador` de su servicio: pyfunc para P1 y P3, y el estimador de scikit-learn
    para P2, que necesita `predict_proba`.
    """
    servicio = servicio_de(question)
    uri = os.getenv(servicio["variable"])
    if uri:
        logger.info(f"Cargando {uri} desde el registry en {mlflow.get_tracking_uri()}")
        return servicio["cargador"](uri), {**metadatos(uri), "origen": "registry"}

    directorio = servicio["directorio"]
    if not (directorio / "MLmodel").exists():
        raise FileNotFoundError(
            f"No hay modelo en {directorio}: corre `python -m mibici_mlops.api.modelo exportar "
            f"--question {question}` con el servidor de MLflow levantado, o define "
            f"{servicio['variable']}"
        )
    info = json.loads((directorio / "version.json").read_text(encoding="utf-8"))
    logger.info(
        f"Cargando la versión {info['version']} de {info['nombre']} empaquetada en la imagen"
    )
    return servicio["cargador"](str(directorio)), {**info, "origen": "imagen"}


Question = Annotated[
    str, typer.Option("--question", help=f"Pregunta cuyo modelo se usa: {' o '.join(SERVICIOS)}.")
]


def servicio_cli(question: str) -> dict:
    try:
        return servicio_de(question)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def exportar(
    question: Question = "p1",
    uri: Annotated[
        str | None,
        typer.Option(help=f"Otro modelo del registry ({FORMATOS_URI}); por defecto, el champion."),
    ] = None,
    destino: Annotated[
        Path | None,
        typer.Option(help="Carpeta destino; por defecto, models/servicio/<pregunta>/."),
    ] = None,
):
    """Descarga del registry el modelo de una pregunta para empaquetarlo en la imagen."""
    servicio = servicio_cli(question)
    uri = uri or servicio["uri"]
    destino = destino or servicio["directorio"]
    info = metadatos(uri)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    # Se descarga con la versión fija aunque se pida por alias: si alguien mueve el alias
    # mientras tanto, el modelo y version.json siguen siendo la misma versión.
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{info['nombre']}/{info['version']}", dst_path=str(destino)
    )
    (destino / "version.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.success(
        f"{uri} (versión {info['version']}, run {info['run_id']}) guardado en {destino}"
    )


@app.command()
def probar(question: Question = "p1"):
    """Carga el modelo de una pregunta como lo hará la API y predice unas filas reales."""
    servicio = servicio_cli(question)
    modelo, info = cargar_modelo(question)
    columnas, objetivo = servicio["columnas"], servicio["objetivo"]
    leer = [*columnas, objetivo] if objetivo else columnas
    ejemplo = pd.read_parquet(servicio["datos"], columns=leer).tail(3)
    ejemplo["prediccion"] = modelo.predict(ejemplo[columnas])
    mostrar = [objetivo, "prediccion"] if objetivo else ["prediccion"]
    if hasattr(modelo, "predict_proba"):  # P2: también la probabilidad de cada clase
        probabilidades = pd.DataFrame(
            modelo.predict_proba(ejemplo[columnas]), columns=modelo.classes_, index=ejemplo.index
        ).add_prefix("p_")
        ejemplo = ejemplo.join(probabilidades)
        mostrar += list(probabilidades.columns)
    logger.success(f"{info}\n{ejemplo[mostrar].round(3).to_string()}")


if __name__ == "__main__":
    app()
