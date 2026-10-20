"""Funciones comunes para registrar los runs de P1, P2 y P3 en MLflow.

Todas las preguntas registran sus runs en el mismo experimento, con la misma versión de los
datos y las mismas etiquetas de equipo, pregunta y commit. Tenerlas en un solo lugar evita que
cada script lo haga distinto.
"""

from pathlib import Path
import subprocess
import urllib.request

from loguru import logger
import mlflow

from mibici_mlops.config import (
    COMANDO_SERVIDOR_MLFLOW,
    DATA_DIR,
    EXPERIMENTO_MLFLOW,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
)

EQUIPO = "43"


def version_datos(ruta: Path = DATA_DIR / "raw.dvc") -> str:
    """md5 que DVC calculó para data/raw (decisión 7 de P2): cambia si cambia cualquier CSV."""
    for linea in ruta.read_text().splitlines():
        if "md5:" in linea:
            return linea.split("md5:")[1].strip()
    raise ValueError(f"No se encontró el md5 en {ruta}")


def estado_git() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin commit: liga cada run con el código que lo produjo."""

    def git(*argumentos: str) -> str:
        resultado = subprocess.run(
            ["git", *argumentos], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        )
        return resultado.stdout.strip()

    return git("rev-parse", "HEAD"), git("status", "--porcelain") != ""


def etiquetas_run(pregunta: str, script: str) -> dict[str, str]:
    """Etiquetas de equipo, pregunta y commit para los runs de `script` (su `__file__`).

    mlflow.user y mlflow.source.name se reemplazan para no registrar el usuario de la computadora
    ni la ruta de su carpeta personal (datos personales).
    """
    commit, cambios_sin_commit = estado_git()
    if cambios_sin_commit:
        logger.warning(
            "Hay cambios sin commit: el commit registrado no es todo el código que corrió"
        )
    return {
        "equipo": EQUIPO,
        "pregunta": pregunta,
        "commit": commit,
        "cambios_sin_commit": str(cambios_sin_commit).lower(),
        "mlflow.user": f"equipo{EQUIPO}",
        "mlflow.source.name": Path(script).resolve().relative_to(PROJ_ROOT).as_posix(),
    }


def verificar_servidor(uri: str = MLFLOW_TRACKING_URI) -> None:
    """Falla de inmediato si el servidor de MLflow no responde.

    Sin esto, MLflow reintenta la conexión durante varios minutos antes de dar un error.
    """
    if not uri.startswith("http"):
        return
    try:
        urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=5)
    except OSError as error:
        raise RuntimeError(
            f"No hay un servidor de MLflow en {uri}. Levántalo en otra terminal, desde la raíz "
            f"del proyecto, con:\n    {COMANDO_SERVIDOR_MLFLOW}"
        ) from error


def usar_experimento() -> None:
    """Apunta MLflow al servidor de tracking y al experimento del equipo; lo crea si no existe."""
    verificar_servidor()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENTO_MLFLOW) is None:
        mlflow.create_experiment(
            EXPERIMENTO_MLFLOW, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
