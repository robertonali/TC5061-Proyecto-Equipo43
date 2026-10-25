"""Lleva el modelo elegido de P2 al Model Registry y le asigna los alias de producción.

Registra el modelo de un run candidato como una versión de P2_MODELO_REGISTRADO, con su linaje
(experimento, run de origen, commit y versiones de los datos) como etiquetas de la versión, y le
asigna el alias `champion`, que marca el modelo en producción. Al siguiente mejor le asigna
`challenger`. Al final comprueba que el modelo en producción se carga por su nombre y alias.

Sin opciones, elige entre los candidatos de P2 (stage_of_work=candidate) los dos de mayor
`val_pr_auc_promedio`, la métrica para elegir modelo (decisiones 8 y 9). Con `--champion` y
`--challenger` se eligen los runs a mano. Si un run ya está registrado, reutiliza su versión.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.modeling.registrar_p2
    python -m mibici_mlops.modeling.registrar_p2 --champion <run_id> --challenger <run_id>
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
import pandas as pd
import typer

from mibici_mlops.config import EXPERIMENTO_MLFLOW, P2_MODELO_REGISTRADO, P2_VENTANAS_PATH
from mibici_mlops.modeling.registro import usar_experimento
from mibici_mlops.modeling.train_p2 import VARIABLES

app = typer.Typer()

METRICA = "val_pr_auc_promedio"

DESCRIPCION_MODELO = """\
P2: riesgo de que una estación de MiBici se quede sin bicicletas (se_vacia) o sin anclajes \
(se_llena) en una ventana de 2 horas. Se predice cada día a las 4:00 para las 12 ventanas del día.

Entrada: una fila por estación y ventana con las columnas de data/processed/p2_ventanas.parquet: \
hora, dia_semana, feriado, capacidad, frec_se_vacia y frec_se_llena (ver mibici_mlops/features.py).
Salida: con mlflow.pyfunc, la clase predicha (sin_riesgo, se_vacia o se_llena); con \
mlflow.sklearn.load_model, predict_proba da la probabilidad de cada clase en el orden de \
classes_ (se_llena, se_vacia, sin_riesgo), útil para ordenar las estaciones por riesgo.

Ciclo de vida con aliases: champion es la versión en producción y challenger la siguiente mejor. \
Cada versión guarda en sus etiquetas el run de origen, el commit y las versiones de los datos. \
Ver references/decisiones_p2.md, decisiones 8 y 9."""


def candidatos_p2(cliente: MlflowClient) -> list[Run]:
    """Runs candidatos de P2 terminados, de mayor a menor `val_pr_auc_promedio`."""
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    return cliente.search_runs(
        [experimento.experiment_id],
        filter_string=(
            "tags.pregunta = 'P2' and tags.stage_of_work = 'candidate' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=[f"metrics.{METRICA} DESC"],
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
        "val_pr_auc_promedio": f"{datos.metrics.get(METRICA, float('nan')):.4f}",
        "test_pr_auc_promedio": f"{datos.metrics.get('test_pr_auc_promedio', float('nan')):.4f}",
    }


def registrar_version(cliente: MlflowClient, run: Run) -> str:
    """Versión del modelo registrado para el run: la crea, o reutiliza la que ya existe."""
    for version in cliente.search_model_versions(f"name = '{P2_MODELO_REGISTRADO}'"):
        if version.tags.get("run_id") == run.info.run_id:
            logger.info(f"{run.info.run_name} ya era la versión {version.version}")
            return version.version
    etiquetas = linaje(run)
    version = mlflow.register_model(modelo_del_run(run), P2_MODELO_REGISTRADO, tags=etiquetas)
    cliente.update_model_version(
        P2_MODELO_REGISTRADO,
        version.version,
        description=(
            f"{run.info.run_name}. PR-AUC promedio de las clases de riesgo: "
            f"{etiquetas['val_pr_auc_promedio']} en validación y "
            f"{etiquetas['test_pr_auc_promedio']} en prueba. Commit {etiquetas['commit'][:7]}, "
            f"manifiesto de datos {etiquetas['version_manifiesto']}."
        ),
    )
    logger.info(f"{run.info.run_name} registrado como versión {version.version}")
    return version.version


def verificar_produccion(cliente: MlflowClient) -> None:
    """Carga el modelo en producción por su nombre y alias y predice unas ventanas reales."""
    uri = f"models:/{P2_MODELO_REGISTRADO}@champion"
    modelo = mlflow.sklearn.load_model(uri)
    ejemplo = pd.read_parquet(P2_VENTANAS_PATH, columns=VARIABLES).tail(3)
    probabilidades = pd.DataFrame(modelo.predict_proba(ejemplo), columns=modelo.classes_)
    clases = mlflow.pyfunc.load_model(uri).predict(ejemplo)
    version = cliente.get_model_version_by_alias(P2_MODELO_REGISTRADO, "champion").version
    logger.success(
        f"{uri} (versión {version}) se carga y predice: {list(clases)}\n"
        f"{probabilidades.round(3).to_string()}"
    )


@app.command()
def main(
    champion: str = typer.Option(
        None, help="Run del modelo en producción (por defecto, el mejor)."
    ),
    challenger: str = typer.Option(
        None, help="Run del siguiente mejor (por defecto, el segundo)."
    ),
):
    usar_experimento()
    cliente = MlflowClient()
    ordenados = candidatos_p2(cliente)
    if champion is None:
        if not ordenados:
            raise RuntimeError("No hay runs candidatos de P2 terminados en el experimento")
        champion = ordenados[0].info.run_id
    if challenger is None:
        restantes = [r.info.run_id for r in ordenados if r.info.run_id != champion]
        challenger = restantes[0] if restantes else None

    try:
        cliente.get_registered_model(P2_MODELO_REGISTRADO)
        cliente.update_registered_model(P2_MODELO_REGISTRADO, description=DESCRIPCION_MODELO)
    except mlflow.exceptions.MlflowException:
        cliente.create_registered_model(
            P2_MODELO_REGISTRADO,
            description=DESCRIPCION_MODELO,
            tags={"equipo": "43", "pregunta": "P2", "question": "P2"},
        )

    for alias, run_id in [("champion", champion), ("challenger", challenger)]:
        if run_id is None:
            continue
        version = registrar_version(cliente, cliente.get_run(run_id))
        cliente.set_registered_model_alias(P2_MODELO_REGISTRADO, alias, version)
        logger.info(f"Alias {alias} → versión {version}")

    verificar_produccion(cliente)


if __name__ == "__main__":
    app()
