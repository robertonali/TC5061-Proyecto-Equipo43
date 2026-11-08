"""Recupera del Model Registry el modelo que sirve la API.

La API sirve dos modelos: el de P1 (demanda por estación y día) y el de P3 (grupo de una
estación). Cada uno se identifica por nombre y alias (SERVICIO_MODELO_URI y SERVICIO_P3_MODELO_URI,
el `champion` de cada pregunta) o por nombre y número de versión (p. ej.
models:/mibici-equipo43-p1-demanda/3). Hay dos formas de llegar a cada uno, y la API usa la
primera que aplique:

1. Si la variable de entorno del modelo está definida (MODELO_URI para P1, MODELO_URI_P3 para
   P3), se carga del registry en el momento de arrancar (p. ej.
   MODELO_URI_P3=models:/mibici-equipo43-p3-clusters@champion con
   MLFLOW_TRACKING_URI=http://host.containers.internal:5000). Requiere el servidor de MLflow.
2. Si no, se carga la copia que `exportar` descargó del registry en models/servicio/<pregunta>/ y
   que se empaqueta en la imagen. Es la opción elegida (ver la sección de despliegue del
   README): el contenedor no depende de que el servidor de MLflow esté corriendo.

Junto al modelo, `exportar` guarda version.json con el nombre, alias, versión y run de origen,
para que la API diga qué versión del registry está sirviendo.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.api.modelo exportar                  # champion de P1
    python -m mibici_mlops.api.modelo exportar --pregunta p3    # champion de P3
    python -m mibici_mlops.api.modelo exportar --uri models:/mibici-equipo43-p1-demanda@challenger
    python -m mibici_mlops.api.modelo exportar --uri models:/mibici-equipo43-p1-demanda/1
    python -m mibici_mlops.api.modelo probar --pregunta p3
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
    P3_PERFILES_PATH,
    SERVICIO_MODELO_DIR,
    SERVICIO_MODELO_URI,
    SERVICIO_P3_MODELO_DIR,
    SERVICIO_P3_MODELO_URI,
)
from mibici_mlops.features import COLUMNAS_P1, COLUMNAS_P3

app = typer.Typer()

# Por pregunta: URI del champion, carpeta de la copia empaquetada y variable de entorno con la que
# se carga del registry en lugar de la copia.
SERVICIOS = {
    "p1": {
        "uri": SERVICIO_MODELO_URI,
        "dir": SERVICIO_MODELO_DIR,
        "variable": "MODELO_URI",
    },
    "p3": {
        "uri": SERVICIO_P3_MODELO_URI,
        "dir": SERVICIO_P3_MODELO_DIR,
        "variable": "MODELO_URI_P3",
    },
}


FORMATOS_URI = "models:/<nombre>@<alias> o models:/<nombre>/<versión>"


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


def servicio(pregunta: str) -> dict:
    """URI, carpeta y variable de entorno del modelo de una pregunta (p1 o p3)."""
    if pregunta not in SERVICIOS:
        raise ValueError(f"La pregunta debe ser {' o '.join(SERVICIOS)}, no {pregunta}")
    return SERVICIOS[pregunta]


def cargar_modelo(pregunta: str = "p1") -> tuple[mlflow.pyfunc.PyFuncModel, dict]:
    """El modelo de una pregunta y sus metadatos (nombre, alias, versión, run de origen)."""
    config = servicio(pregunta)
    carpeta = config["dir"]
    uri = os.getenv(config["variable"])
    if uri:
        logger.info(f"Cargando {uri} desde el registry en {mlflow.get_tracking_uri()}")
        return mlflow.pyfunc.load_model(uri), {**metadatos(uri), "origen": "registry"}

    if not (carpeta / "MLmodel").exists():
        raise FileNotFoundError(
            f"No hay modelo en {carpeta}: corre `python -m mibici_mlops.api.modelo exportar "
            f"--pregunta {pregunta}` con el servidor de MLflow levantado, o define "
            f"{config['variable']}"
        )
    info = json.loads((carpeta / "version.json").read_text(encoding="utf-8"))
    logger.info(
        f"Cargando la versión {info['version']} de {info['nombre']} empaquetada en la imagen"
    )
    return mlflow.pyfunc.load_model(str(carpeta)), {**info, "origen": "imagen"}


@app.command()
def exportar(
    pregunta: Annotated[str, typer.Option(help="Modelo a exportar: p1 o p3.")] = "p1",
    uri: Annotated[
        str | None, typer.Option(help=f"Modelo del registry: {FORMATOS_URI}. Por defecto, champion.")
    ] = None,
    destino: Annotated[
        Path | None, typer.Option(help="Carpeta de destino. Por defecto, models/servicio/<pregunta>.")
    ] = None,
):
    """Descarga del registry el modelo que se empaqueta en la imagen."""
    config = servicio(pregunta)
    uri = uri or config["uri"]
    destino = destino or config["dir"]
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
def probar(pregunta: Annotated[str, typer.Option(help="Modelo a probar: p1 o p3.")] = "p1"):
    """Carga el modelo como lo hará la API y predice unas filas reales."""
    modelo, info = cargar_modelo(pregunta)
    if pregunta == "p3":
        ejemplo = pd.read_parquet(P3_PERFILES_PATH, columns=["estacion", *COLUMNAS_P3]).tail(3)
        ejemplo["cluster"] = modelo.predict(ejemplo[COLUMNAS_P3].astype("float64"))
        logger.success(f"{info}\n{ejemplo[['estacion', 'cluster']].to_string(index=False)}")
        return
    ejemplo = pd.read_parquet(P1_FEATURES_PATH, columns=[*COLUMNAS_P1, "viajes"]).tail(3)
    ejemplo["prediccion"] = modelo.predict(ejemplo[COLUMNAS_P1])
    logger.success(f"{info}\n{ejemplo[['viajes', 'prediccion']].round(2).to_string()}")


if __name__ == "__main__":
    app()
