"""Lleva el modelo elegido de P1 al Model Registry y le asigna los alias de producción.

Registra el modelo de un run candidato como una versión de P1_MODELO_REGISTRADO, con su linaje
(experimento, run de origen, commit y versiones de los datos) como etiquetas de la versión, y le
asigna el alias `champion`, que marca el modelo en producción. Al siguiente mejor le asigna
`challenger`. Al final comprueba que el modelo en producción se carga por su nombre y alias.

Sin opciones, elige entre los candidatos de P1 (stage_of_work=candidate) los dos de menor
`val_mae`: el error en viajes por estación-día es lo que cuesta al operador al repartir
bicicletas y personal. Con `--champion` y `--challenger` se eligen los runs a mano. Si un run ya
está registrado, reutiliza su versión.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.modeling.registrar_p1
    python -m mibici_mlops.modeling.registrar_p1 --champion <run_id> --challenger <run_id>
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
import pandas as pd
import typer

from mibici_mlops.config import EXPERIMENTO_MLFLOW, P1_FEATURES_PATH, P1_MODELO_REGISTRADO
from mibici_mlops.features import COLUMNAS_P1
from mibici_mlops.modeling.registro import etiquetar_run_de_proyecto, usar_experimento

app = typer.Typer()

METRICA = "val_mae"

DESCRIPCION_MODELO = """\
P1: número de viajes que iniciarán en cada estación de MiBici durante el día siguiente. Se \
predice cada día al cierre (23:59) para el día siguiente; operación lo usa para repartir \
bicicletas y personal por zona y por día.

Entrada: una fila por estación y día con las columnas de data/processed/p1_estacion_dia.parquet: \
dia_semana, mes, es_fin_de_semana, es_feriado, capacidad, lat, lon, lag_7d y media_7d (ver \
mibici_mlops/features.py). Salida: viajes esperados (recortar a 0 los negativos).

Ciclo de vida con aliases: champion es la versión en producción y challenger la siguiente mejor. \
Cada versión guarda en sus etiquetas el run de origen, el commit y las versiones de los datos. \
Ver references/decisiones_p1.md."""


def candidatos_p1(cliente: MlflowClient) -> list[Run]:
    """Runs candidatos de P1 terminados, de menor a mayor `val_mae`."""
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    return cliente.search_runs(
        [experimento.experiment_id],
        filter_string=(
            "tags.pregunta = 'P1' and tags.stage_of_work = 'candidate' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=[f"metrics.{METRICA} ASC"],
    )


def modelo_del_run(run: Run) -> str:
    """URI del modelo que guardó el run con `log_model` (models:/<id>)."""
    modelos = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        output_format="list",
    )
    if not modelos:
        raise ValueError(f"El run {run.info.run_id} no guardó un modelo (¿es un baseline?)")
    return f"models:/{modelos[0].model_id}"


def linaje(run: Run) -> dict[str, str]:
    """Etiquetas de la versión: de dónde salió el modelo y con qué datos y código."""
    datos = run.data
    return {
        "experimento": EXPERIMENTO_MLFLOW,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "commit": datos.tags.get("commit", ""),
        "cambios_sin_commit": datos.tags.get("cambios_sin_commit", ""),
        "version_manifiesto": datos.params.get("version_manifiesto", ""),
        "version_datos": datos.params.get("version_datos", ""),
        "model_family": datos.tags.get("model_family", ""),
        "val_mae": f"{datos.metrics.get(METRICA, float('nan')):.4f}",
        "test_mae": f"{datos.metrics.get('test_mae', float('nan')):.4f}",
        "test_mae_mejora_vs_baseline": (
            f"{datos.metrics.get('test_mae_mejora_vs_baseline', float('nan')):.4f}"
        ),
    }


def registrar_version(cliente: MlflowClient, run: Run) -> str:
    """Versión del modelo registrado para el run: la crea, o reutiliza la que ya existe."""
    for version in cliente.search_model_versions(f"name = '{P1_MODELO_REGISTRADO}'"):
        if version.tags.get("run_id") == run.info.run_id:
            logger.info(f"{run.info.run_name} ya era la versión {version.version}")
            return version.version
    etiquetas = linaje(run)
    version = mlflow.register_model(modelo_del_run(run), P1_MODELO_REGISTRADO, tags=etiquetas)
    cliente.update_model_version(
        P1_MODELO_REGISTRADO,
        version.version,
        description=(
            f"{run.info.run_name}. MAE (viajes por estación-día): {etiquetas['val_mae']} en "
            f"validación y {etiquetas['test_mae']} en prueba. Commit {etiquetas['commit'][:7]}, "
            f"manifiesto de datos {etiquetas['version_manifiesto']}."
        ),
    )
    logger.info(f"{run.info.run_name} registrado como versión {version.version}")
    return version.version


def verificar_produccion(cliente: MlflowClient) -> None:
    """Carga el modelo en producción por su nombre y alias y predice unas filas reales."""
    uri = f"models:/{P1_MODELO_REGISTRADO}@champion"
    ejemplo = pd.read_parquet(P1_FEATURES_PATH, columns=COLUMNAS_P1).tail(3)
    viajes = mlflow.pyfunc.load_model(uri).predict(ejemplo)
    version = cliente.get_model_version_by_alias(P1_MODELO_REGISTRADO, "champion").version
    logger.success(f"{uri} (versión {version}) se carga y predice: {list(viajes.round(1))}")


@app.command()
def main(
    champion: str = typer.Option(
        None, help="Run del modelo en producción (por defecto, el mejor)."
    ),
    challenger: str = typer.Option(
        None, help="Run del siguiente mejor (por defecto, el segundo)."
    ),
):
    etiquetar_run_de_proyecto("P1", __file__, "register")
    usar_experimento()
    cliente = MlflowClient()
    ordenados = candidatos_p1(cliente)
    if champion is None:
        if not ordenados:
            raise RuntimeError("No hay runs candidatos de P1 terminados en el experimento")
        champion = ordenados[0].info.run_id
    if challenger is None:
        restantes = [r.info.run_id for r in ordenados if r.info.run_id != champion]
        challenger = restantes[0] if restantes else None

    try:
        cliente.get_registered_model(P1_MODELO_REGISTRADO)
        cliente.update_registered_model(P1_MODELO_REGISTRADO, description=DESCRIPCION_MODELO)
    except mlflow.exceptions.MlflowException:
        cliente.create_registered_model(
            P1_MODELO_REGISTRADO,
            description=DESCRIPCION_MODELO,
            tags={"equipo": "43", "pregunta": "P1", "question": "P1"},
        )

    for alias, run_id in [("champion", champion), ("challenger", challenger)]:
        if run_id is None:
            continue
        version = registrar_version(cliente, cliente.get_run(run_id))
        cliente.set_registered_model_alias(P1_MODELO_REGISTRADO, alias, version)
        logger.info(f"Alias {alias} → versión {version}")

    verificar_produccion(cliente)


if __name__ == "__main__":
    app()
