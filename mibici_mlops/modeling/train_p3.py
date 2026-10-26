from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.base import clone
from loguru import logger
from pathlib import Path
import tempfile
import pandas as pd
import joblib
import mlflow
import typer

from mibici_mlops.config import (
    ESTACIONES_PATH,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    P3_CLUSTERS_PATH,
    P3_GRUPOS_BUSQUEDA,
    P3_MODEL_PATH,
    P3_PERFILES_PATH,
    RANDOM_STATE,
)
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
from mibici_mlops.plots import p3 as generar_plots

app = typer.Typer()


PARAMETROS = {
    'kmeans_k3': {
        'n_clusters': 3,
        'n_init': 10
    },
    'kmeans_k4': {
        'n_clusters': 4,
        'n_init': 10
    },
    'gmm_diag_k2': {
        'n_components': 2,
        'n_init': 5,
        'covariance_type': 'diag'
    },
}

# Candidatos: se agrupan por el perfil de uso. Son los que compara scripts/run_p3.py y entre los
# que registrar_p3 elige el champion.
MODELOS = {
    'kmeans_k3': KMeans,
    'kmeans_k4': KMeans,
    'gmm_diag_k2': GaussianMixture,
}

# Baseline (decisión 6 de P3): KMeans con 3 grupos sobre latitud y longitud estandarizadas, es
# decir, zonas de la ciudad. Se evalúa en el espacio del perfil de uso, igual que los candidatos.
BASELINE = 'zonas_k3'
PARAMETROS_BASELINE = {'n_clusters': 3, 'n_init': 10}

# Búsqueda del número de grupos de KMeans (decisión 5), con un run hijo por número de grupos.
BUSQUEDA = 'busqueda-kmeans'

FAMILIAS = {KMeans: 'kmeans', GaussianMixture: 'gaussian_mixture'}


class P3Trainer:
    def __init__(self, perfiles, estaciones, modelo, por_coordenadas=False):
        self.perfiles = perfiles.copy()
        self.estaciones = estaciones[['id', 'latitude', 'longitude']].rename(columns={
            'id': 'estacion',
            'latitude': 'estacion_lat',
            'longitude': 'estacion_lon',
        })
        self.ids_estaciones_train = self.perfiles[self.perfiles.conjunto=='entrenamiento'].estacion.to_numpy()
        self.ids_estaciones_val = self.perfiles[self.perfiles.conjunto=='validacion'].estacion.to_numpy()
        self.ids_estaciones_test = self.perfiles[self.perfiles.conjunto=='prueba'].estacion.to_numpy()

        conjunto = self.perfiles.conjunto
        columnas_perfil = [c for c in self.perfiles.columns if c.startswith(('sal_', 'lle_'))]
        X_perfiles = self.perfiles[columnas_perfil].astype('float64')

        # Las métricas siempre se calculan sobre el perfil de uso (self.X*).
        self.X = X_perfiles.loc[conjunto == 'entrenamiento']
        self.X_val = X_perfiles.loc[conjunto == 'validacion']
        self.X_test = X_perfiles.loc[conjunto == 'prueba']

        # Variables con que se ajusta el modelo: el perfil, o las coordenadas en el baseline.
        if por_coordenadas:
            coordenadas = self.perfiles[['estacion']].merge(
                self.estaciones, on='estacion', how='left', validate='many_to_one'
            )[['estacion_lat', 'estacion_lon']]
            coordenadas.index = self.perfiles.index
            ajuste = coordenadas
        else:
            ajuste = X_perfiles
        self.Z = ajuste.loc[conjunto == 'entrenamiento']
        self.Z_val = ajuste.loc[conjunto == 'validacion']
        self.Z_test = ajuste.loc[conjunto == 'prueba']

        self.modelo = clone(modelo)

    def fit_predict(self):
        self.etiquetas = self.modelo.fit_predict(self.Z)
        self.etiquetas_val = self.modelo.predict(self.Z_val)
        self.etiquetas_test = self.modelo.predict(self.Z_test)

        self.asignaciones = pd.concat(
            [
                pd.DataFrame({
                    'estacion': self.ids_estaciones_train,
                    'conjunto': 'entrenamiento',
                    'cluster': self.etiquetas
                }),
                pd.DataFrame({
                    'estacion': self.ids_estaciones_val,
                    'conjunto': 'validacion',
                    'cluster': self.etiquetas_val
                }),
                pd.DataFrame({
                    'estacion': self.ids_estaciones_test,
                    'conjunto': 'prueba',
                    'cluster': self.etiquetas_test
                })
            ],
            ignore_index=True,
        )

        # Añadir coordenadas a las asignaciones
        self.asignaciones = self.asignaciones.merge(
            self.estaciones,
            on='estacion',
            how='left',
            validate='many_to_one',
        )

        n_clusters = len(set(self.etiquetas))
        if not 1 < n_clusters < len(self.X):
            raise ValueError(
                'Silhouette requiere al menos 2 clusters y menos clusters que filas.'
            )

        grupos_train = pd.Series(
            self.etiquetas,
            index=self.ids_estaciones_train,
        )
        grupos_val = pd.Series(
            self.etiquetas_val,
            index=self.ids_estaciones_val,
        )

        estaciones_comunes = grupos_train.index.intersection(grupos_val.index)
        if len(estaciones_comunes) < 2:
            raise ValueError('Se necesitan al menos dos estaciones comunes para calcular ARI.')
        ari =  adjusted_rand_score(
            grupos_train.loc[estaciones_comunes],
            grupos_val.loc[estaciones_comunes],
        )

        self.metricas = {
            'ari_estabilidad': float(ari),
        }
        # Métricas internas en el espacio del perfil (decisión 6), en cada conjunto.
        for prefijo, X, etiquetas in [
            ('train', self.X, self.etiquetas),
            ('val', self.X_val, self.etiquetas_val),
            ('test', self.X_test, self.etiquetas_test),
        ]:
            if 1 < len(set(etiquetas)) < len(X):
                self.metricas[f'{prefijo}_silhouette'] = float(silhouette_score(X, etiquetas))
                self.metricas[f'{prefijo}_davies_bouldin'] = float(davies_bouldin_score(X, etiquetas))
                self.metricas[f'{prefijo}_calinski_harabasz'] = float(
                    calinski_harabasz_score(X, etiquetas)
                )

        if hasattr(self.modelo, 'inertia_'):
            self.metricas['inertia'] = float(self.modelo.inertia_)

        if hasattr(self.modelo, 'bic'):
            self.metricas['bic'] = float(self.modelo.bic(self.Z))

        return self.asignaciones, self.metricas

    def save(self, model_path, output_path):
        joblib.dump(self.modelo, model_path)
        self.asignaciones.to_parquet(
            output_path,
            index=False,
        )


def parametros_comunes(perfiles_path: Path, p3: P3Trainer) -> dict:
    return {
        'fecha_inicio': FECHA_INICIO,
        'fecha_fin': FECHA_FIN,
        'fecha_inicio_validacion': FECHA_INICIO_VALIDACION,
        'fecha_inicio_prueba': FECHA_INICIO_PRUEBA,
        'version_datos': version_datos(),
        'tabla': ruta_para_registro(perfiles_path),
        'estaciones_entrenamiento': len(p3.X),
        'estaciones_validacion': len(p3.X_val),
        'estaciones_prueba': len(p3.X_test),
    }


def registrar_datos(p3: P3Trainer, fuente: str) -> None:
    '''Manifiesto de la ingesta y huella de cada conjunto de perfiles en data/.

    Los perfiles son fracciones de viajes por estación: no tienen datos de las personas usuarias.
    '''
    registrar_manifiesto()
    registrar_conjuntos(
        {'training': p3.Z, 'validation': p3.Z_val, 'testing': p3.Z_test},
        fuente,
        'p3_perfiles',
        None,
    )


def buscar_kmeans(perfiles, estaciones, random_state, parametros_fuente, tags):
    '''Búsqueda del número de grupos de KMeans (P3_GRUPOS_BUSQUEDA) con nested runs.

    El run padre guarda el mejor número de grupos por silhouette de entrenamiento (decisión 5)
    y la tabla de resultados; cada número de grupos es un run hijo.
    '''
    resultados = []
    with mlflow.start_run(run_name='p3-busqueda-kmeans') as run:
        mlflow.set_tags({**tags, 'stage_of_work': 'search', 'model_family': 'kmeans'})
        mlflow.set_tag('mlflow.runName', 'p3-busqueda-kmeans')
        primero = None
        for k in P3_GRUPOS_BUSQUEDA:
            parametros = {'n_clusters': k, 'n_init': 10, 'random_state': random_state}
            p3 = P3Trainer(perfiles, estaciones, KMeans(**parametros))
            p3.fit_predict()
            primero = primero or p3
            with mlflow.start_run(run_name=f'p3-kmeans-k{k}', nested=True):
                mlflow.set_tags({**tags, 'stage_of_work': 'search', 'model_family': 'kmeans'})
                mlflow.log_params({'modelo': f'kmeans_k{k}', **parametros})
                mlflow.log_metrics(p3.metricas)
            # Curva del padre: un paso por número de grupos.
            mlflow.log_metric('train_silhouette_busqueda', p3.metricas['train_silhouette'], step=k)
            resultados.append({'n_clusters': k, **p3.metricas})

        tabla = pd.DataFrame(resultados).sort_values('train_silhouette', ascending=False)
        mejor = int(tabla['n_clusters'].iloc[0])
        registrar_parametros({
            **parametros_comunes(parametros_fuente, primero),
            'modelo': BUSQUEDA,
            'grupos_busqueda': f'{min(P3_GRUPOS_BUSQUEDA)}-{max(P3_GRUPOS_BUSQUEDA)}',
            'mejor_n_clusters': mejor,
        })
        mlflow.log_metric('mejor_train_silhouette', tabla['train_silhouette'].iloc[0])
        mlflow.log_text(tabla.to_csv(index=False), 'busqueda/resultados.csv')
        registrar_manifiesto()
        registrar_configuracion({'modelo': BUSQUEDA, 'grupos': list(P3_GRUPOS_BUSQUEDA)}, 'P3')
    logger.success(f'Mejor número de grupos por silhouette: {mejor}')
    return run.info.run_id


@app.command()
def p3train(
    modelo: str = typer.Option(
        'kmeans_k3', help=f'Modelo a correr: {', '.join([*MODELOS, BASELINE, BUSQUEDA])}.'
    ),
    input_path : Path = P3_PERFILES_PATH,
    stations_path : Path = ESTACIONES_PATH,
    output_path : Path = P3_CLUSTERS_PATH,
    model_path : Path = P3_MODEL_PATH,
    random_state : int = RANDOM_STATE,
    run_id : str = None
):
    if modelo not in [*MODELOS, BASELINE, BUSQUEDA]:
        raise typer.BadParameter(
            f'usa uno de: {', '.join([*MODELOS, BASELINE, BUSQUEDA])}', param_hint='--modelo'
        )
    perfiles = pd.read_parquet(input_path)
    estaciones = pd.read_csv(stations_path)

    usar_experimento()
    tags = etiquetas_run("P3", __file__)

    if modelo == BUSQUEDA:
        return buscar_kmeans(perfiles, estaciones, random_state, input_path, tags)

    es_baseline = modelo == BASELINE
    if es_baseline:
        parametros = {**PARAMETROS_BASELINE, 'random_state': random_state}
        instancia_modelo = make_pipeline(StandardScaler(), KMeans(**parametros))
        familia = 'kmeans'
    else:
        parametros = {**PARAMETROS[modelo], 'random_state': random_state}
        instancia_modelo = MODELOS[modelo](**parametros)
        familia = FAMILIAS[MODELOS[modelo]]
    p3 = P3Trainer(perfiles, estaciones, instancia_modelo, por_coordenadas=es_baseline)
    p3.fit_predict()

    p3.save(model_path, output_path)
    logger.success(f'Modelo entrenado, datos guardados en {output_path}')
    logger.info('Métricas: {}', p3.metricas)

    tags = {
        **tags,
        "stage_of_work": "baseline" if es_baseline else "candidate",
        "model_family": familia,
    }
    variables = 'estacion_lat,estacion_lon' if es_baseline else 'perfil_de_uso'

    with mlflow.start_run(
        run_id=run_id,
        run_name=f"p3-{modelo}",
    ) as run:
        logger.info('MLflow run ID: {}', run.info.run_id)
        mlflow.set_tags(tags)
        mlflow.set_tag('mlflow.runName', f'p3-{modelo}')
        registrar_parametros({
            **parametros_comunes(input_path, p3),
            'modelo': modelo,
            'variables': variables,
            **parametros,
        })
        mlflow.log_metrics(p3.metricas)
        registrar_datos(p3, ruta_para_registro(input_path))
        registrar_configuracion(
            {'modelo': modelo, 'hiperparametros': parametros, 'variables': variables}, 'P3'
        )

        if not es_baseline:
            mlflow.sklearn.log_model(
                p3.modelo,
                name=modelo,
                input_example=p3.X.head(3),
                signature=infer_signature(p3.X, p3.etiquetas),
                serialization_format="skops",
            )

        # En una carpeta temporal: reports/figures/p3 está versionado y solo lo actualiza el
        # champion (scripts/run_p3.py); si no, cada run deja el repositorio con cambios.
        with tempfile.TemporaryDirectory() as carpeta:
            generar_plots(
                clusters_path=output_path,
                profiles_path=input_path,
                output_dir=Path(carpeta),
                run_id=run.info.run_id,
            )

    return run.info.run_id


if __name__ == '__main__':
    app()
