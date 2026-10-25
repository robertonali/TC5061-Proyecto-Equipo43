'''
Lleva el modelo elegido de P3 al Model Registry y le asigna los alias de producción.

Registra el modelo de un run candidato como una versión de P3_MODELO_REGISTRADO, con su linaje
(experimento, run de origen, commit y versiones de los datos) como etiquetas de la versión, y le
asigna el alias `champion`, que marca el modelo en producción. Al siguiente mejor le asigna
`challenger`. Al final comprueba que el modelo en producción se carga por su nombre y alias.

Sin opciones, se ordena la lista de modelos por ARI, siendo el primero el `champion` y el
segundo, si hay, el `challenger`.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.modeling.registrar_p3
    python -m mibici_mlops.modeling.registrar_p3 --champion <run_id> --challenger <run_id>
'''

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
import pandas as pd
import typer

from mibici_mlops.config import EXPERIMENTO_MLFLOW, P3_MODELO_REGISTRADO, P3_PERFILES_PATH
from mibici_mlops.modeling.registro import usar_experimento

app = typer.Typer()

METRICA = 'ari_estabilidad'

DESCRIPCION_MODELO = '''
P3: Modelo de clustering que agrupa las estaciones en base a perfiles, los perfiles que describen a cada
grupo se pueden encontrar en `reports/figures/p3/p3_perfiles_por_grupo.csv` y un mapa que muestra
la ubicación de cada estación y el grupo al que pertenece en `reports/figures/p3/p3_clusters.html`.
'''


def candidatos_p3(cliente: MlflowClient) -> list[Run]:
    '''Runs candidatos de P3 terminados, de mayor a menor `ari_estabilidad`.'''
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    return cliente.search_runs(
        [experimento.experiment_id],
        filter_string="tags.pregunta = 'P3' and tags.stage_of_work = 'candidate' and attributes.status = 'FINISHED'",
        order_by=[f'metrics.{METRICA} DESC'],
    )


def modelo_del_run(run: Run) -> str:
    '''URI del modelo que guardó el run con `log_model` (models:/<id>).'''
    modelos = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        output_format='list',
    )
    if not modelos:
        raise ValueError(f'El run {run.info.run_id} no guardó un modelo (¿es un baseline?)')
    return f'models:/{modelos[0].model_id}'


def linaje(run: Run) -> dict[str, str]:
    '''Etiquetas de la versión: de dónde salió el modelo y con qué datos y código.'''
    datos = run.data
    return {
        'experimento': EXPERIMENTO_MLFLOW,
        'run_id': run.info.run_id,
        'run_name': run.info.run_name,
        'commit': datos.tags.get('commit', ''),
        'cambios_sin_commit': datos.tags.get('cambios_sin_commit', ''),
        'model_family': datos.tags.get('model_family', ''),
        'ari_estabilidad': f'{datos.metrics.get(METRICA, float('nan')):.4f}',
        'val_silhouette': f'{datos.metrics.get('val_silhouette', float('nan')):.4f}',
    }


def registrar_version(cliente: MlflowClient, run: Run) -> str:
    '''Versión del modelo registrado para el run: la crea, o reutiliza la que ya existe.'''
    for version in cliente.search_model_versions(f"name = '{P3_MODELO_REGISTRADO}'"):
        if version.tags.get('run_id') == run.info.run_id:
            logger.info(f'{run.info.run_name} ya era la versión {version.version}')
            return version.version
    etiquetas = linaje(run)
    version = mlflow.register_model(modelo_del_run(run), P3_MODELO_REGISTRADO, tags=etiquetas)
    cliente.update_model_version(
        P3_MODELO_REGISTRADO,
        version.version,
        description=(
            f'{run.info.run_name}.  Metricas: '
            f'{etiquetas['ari_estabilidad']} en validación y '
            f'{etiquetas['val_silhouette']} en prueba. Commit {etiquetas['commit'][:7]}.'
        ),
    )
    logger.info(f'{run.info.run_name} registrado como versión {version.version}')
    return version.version


def verificar_produccion(cliente: MlflowClient) -> None:
    '''Carga el modelo en producción por su nombre y alias y predice clusters.'''
    uri = f"models:/{P3_MODELO_REGISTRADO}@champion"
    modelo = mlflow.pyfunc.load_model(uri)

    perfiles = pd.read_parquet(P3_PERFILES_PATH)
    columnas = [
        c for c in perfiles.columns
        if c.startswith(("sal_", "lle_"))
    ]
    muestra = perfiles.loc[
        perfiles["conjunto"].eq("validacion")
    ].head(3)

    if muestra.empty:
        raise ValueError("No hay perfiles de validación para verificar el modelo.")

    predicciones = modelo.predict(muestra[columnas])
    version = cliente.get_model_version_by_alias(
        P3_MODELO_REGISTRADO, "champion"
    ).version

    asignaciones = list(zip(muestra["estacion"], predicciones))
    logger.success(
        f"{uri} (versión {version}) predice estación → cluster: {asignaciones}"
    )


@app.command()
def main(
    champion: str = typer.Option(
        None, help='Run del modelo en producción (por defecto, el mejor).'
    ),
    challenger: str = typer.Option(
        None, help='Run del siguiente mejor (por defecto, el segundo).'
    ),
):
    usar_experimento()
    cliente = MlflowClient()
    ordenados = candidatos_p3(cliente)
    if champion is None:
        if not ordenados:
            raise RuntimeError('No hay runs candidatos de P3 terminados en el experimento')
        champion = ordenados[0].info.run_id
    if challenger is None:
        restantes = [r.info.run_id for r in ordenados if r.info.run_id != champion]
        challenger = restantes[0] if restantes else None

    try:
        cliente.get_registered_model(P3_MODELO_REGISTRADO)
        cliente.update_registered_model(P3_MODELO_REGISTRADO, description=DESCRIPCION_MODELO)
    except mlflow.exceptions.MlflowException:
        cliente.create_registered_model(
            P3_MODELO_REGISTRADO,
            description=DESCRIPCION_MODELO,
            tags={'equipo': '43', 'pregunta': 'P3', 'question': 'P3'},
        )

    for alias, run_id in [('champion', champion), ('challenger', challenger)]:
        if run_id is None:
            continue
        version = registrar_version(cliente, cliente.get_run(run_id))
        cliente.set_registered_model_alias(P3_MODELO_REGISTRADO, alias, version)
        logger.info(f'Alias {alias} → versión {version}')

    verificar_produccion(cliente)


if __name__ == '__main__':
    app()
