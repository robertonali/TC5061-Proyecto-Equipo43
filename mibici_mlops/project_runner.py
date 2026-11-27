"""Punto de entrada `pipeline` del MLproject: encadena ingesta, features, entrenamiento y registro.
`mlflow run . -e pipeline --env-manager local --experiment-name mibici-equipo43` ejecuta, en orden y
cada uno como un `mlflow run` del mismo proyecto, los pasos del README: `ingest`, y por pregunta
`features`, los `train` de sus modelos y `register`, con el champion y el challenger que
documentan las decisiones de cada pregunta (ELEGIDOS). Cada paso queda como un run en el servidor.
Uso directo:
    python -m mibici_mlops.project_runner --preguntas p2 --descarga sin-descargar
"""

import os

from loguru import logger
import mlflow
import typer

app = typer.Typer()

EXPERIMENTO = "mibici-equipo43"

# Pregunta → (modelo, parámetros extra) en el orden de los comandos del README.
ENTRENAMIENTOS = {
    "p1": [
        ("baseline", {}),
        ("ridge", {}),
        ("rf", {}),
        ("hgb", {}),
        ("busqueda-hgb", {}),
        ("hgb", {"max_leaf_nodes": 127}),
    ],
    "p2": [
        ("mayoritaria", {}),
        ("frecuencia", {}),
        ("logistica", {}),
        ("hgb", {}),
        ("hgb", {"class_weight": "none"}),
        ("busqueda-hgb", {}),
        ("hgb", {"learning_rate": 0.05}),
    ],
    "p3": [
        ("zonas_k3", {}),
        ("busqueda-kmeans", {}),
        ("kmeans_k3", {}),
        ("kmeans_k4", {}),
        ("gmm_diag_k2", {}),
    ],
}


# Pregunta → (run del champion, run del challenger), por nombre de run (decisiones de P1, P2 y P3).
# Si se cambia el modelo en producción, se cambia aquí y en las decisiones de la pregunta.
ELEGIDOS = {
    "p1": ("p1-hgb-poisson-lr0.1-hojas63-arboles300", "p1-rf-arboles100-prof16-hoja20"),
    "p2": ("p2-hgb-balanceado-lr0.05-hojas31-arboles300", "p2-logistica-balanceado-c1.0"),
    "p3": ("p3-kmeans_k3", "p3-gmm_diag_k2"),
}


def run_id_de(pregunta: str, nombre: str) -> str:
    """Run candidato terminado más reciente de la pregunta con ese nombre."""
    cliente = mlflow.MlflowClient()
    experimento = cliente.get_experiment_by_name(EXPERIMENTO)
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string=(
            f"tags.pregunta = '{pregunta.upper()}' and tags.stage_of_work = 'candidate' "
            f"and attributes.status = 'FINISHED' and attributes.run_name = '{nombre}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No hay un run terminado de {pregunta} llamado {nombre!r}")
    return runs[0].info.run_id


def paso(entry_point: str, parametros: dict) -> None:
    """Corre un entry point del MLproject como run del experimento y falla si no termina bien."""
    logger.info(f"mlflow run . -e {entry_point} {parametros}")
    mlflow.projects.run(
        uri=".",
        entry_point=entry_point,
        parameters=parametros,
        experiment_name=EXPERIMENTO,
        env_manager="local",
        synchronous=True,
    )


@app.command()
def main(
    preguntas: str = typer.Option("p1,p2,p3", help="Preguntas a correr, separadas por comas."),
    descarga: str = typer.Option(
        "descargar", help="`descargar` o `sin-descargar` (solo data/raw)."
    ),
):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise typer.BadParameter("define MLFLOW_TRACKING_URI con el servidor de tracking")
    lista = [p.strip() for p in preguntas.split(",") if p.strip()]
    desconocidas = [p for p in lista if p not in ENTRENAMIENTOS]
    if desconocidas:
        raise typer.BadParameter(
            f"preguntas desconocidas: {desconocidas}", param_hint="--preguntas"
        )
    paso("ingest", {"descarga": descarga})
    for pregunta in lista:
        paso("features", {"question": pregunta})
        for modelo, extra in ENTRENAMIENTOS[pregunta]:
            paso("train", {"question": pregunta, "model": modelo, **extra})
        champion, challenger = (run_id_de(pregunta, n) for n in ELEGIDOS[pregunta])
        paso("register", {"question": pregunta, "champion": champion, "challenger": challenger})


if __name__ == "__main__":
    app()
