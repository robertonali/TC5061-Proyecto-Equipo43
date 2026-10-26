from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.base import clone
from loguru import logger
from pathlib import Path
from ast import Dict
import pandas as pd
import joblib
import mlflow
import typer

from mibici_mlops.config import ESTACIONES_PATH, P3_CLUSTERS_PATH, P3_MODEL_PATH, P3_PERFILES_PATH, RANDOM_STATE
from mibici_mlops.modeling.registro import usar_experimento, etiquetas_run
from mibici_mlops.plots import p3 as generar_plots

app = typer.Typer()


PARAMETROS = {
    'kmeans_k3': {
        'n_clusters': 3,
        'n_init': 10
    },
    'gmm_diag_k2': {
        'n_components': 2,
        'n_init': 5,
        'covariance_type': 'diag'
    },
}

MODELOS = {
    'kmeans_k3': KMeans,
    'gmm_diag_k2': GaussianMixture,
}

class P3Trainer:
    def __init__(self, perfiles, estaciones, modelo):
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

        self.X = X_perfiles.loc[conjunto == 'entrenamiento']
        self.X_val = X_perfiles.loc[conjunto == 'validacion']
        self.X_test = X_perfiles.loc[conjunto == 'prueba']

        self.modelo = clone(modelo)

    def fit_predict(self):
        self.etiquetas = self.modelo.fit_predict(self.X)
        self.etiquetas_val = self.modelo.predict(self.X_val)
        self.etiquetas_test = self.modelo.predict(self.X_test)

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
            'train_silhouette': float(silhouette_score(self.X, self.etiquetas)),
            'ari_estabilidad': float(ari),
        }

        n_clusters_val = len(set(self.etiquetas_val))
        if 1 < n_clusters_val < len(self.X_val):
            self.metricas['val_silhouette'] = float(
                silhouette_score(self.X_val, self.etiquetas_val)
            )

        if hasattr(self.modelo, 'inertia_'):
            self.metricas['inertia'] = float(self.modelo.inertia_)

        if hasattr(self.modelo, 'bic'):
            self.metricas['bic'] = float(self.modelo.bic(self.X))

        return self.asignaciones, self.metricas

    def save(self, model_path, output_path):
        joblib.dump(self.modelo, model_path)
        self.asignaciones.to_parquet(
            output_path,
            index=False,
        )

@app.command()
def p3train(
    modelo: str = typer.Option('kmeans_k3', help=f'Modelo a correr: {', '.join(MODELOS)}.'),
    input_path : Path = P3_PERFILES_PATH,
    stations_path : Path = ESTACIONES_PATH,
    output_path : Path = P3_CLUSTERS_PATH,
    model_path : Path = P3_MODEL_PATH,
    random_state : int = RANDOM_STATE,
    run_id : str = None
):
    if modelo not in MODELOS:
        raise typer.BadParameter(
            f'usa uno de: {', '.join(MODELOS)}', param_hint='--modelo'
        )
    perfiles = pd.read_parquet(input_path)
    estaciones = pd.read_csv(stations_path)
    parametros = {
        **PARAMETROS[modelo],
        'random_state': random_state
    }
    instancia_modelo = MODELOS[modelo](**parametros)
    p3 = P3Trainer(perfiles, estaciones, instancia_modelo)
    p3.fit_predict()

    p3.save(model_path, output_path)
    logger.success(f'Modelo entrenado, datos guardados en {output_path}')
    logger.info('Métricas: {}', p3.metricas)


    usar_experimento()
    tags = {
        **etiquetas_run("P3", __file__),
        "stage_of_work": "candidate",
        "model_family": type(instancia_modelo).__name__,
    }


    with mlflow.start_run(
        run_id=run_id,
        run_name=f"p3-{modelo}",
    ) as run:
        logger.info('MLflow run ID: {}', run.info.run_id)
        mlflow.set_tags(tags)
        mlflow.log_params({"modelo": modelo, **parametros})
        mlflow.log_metrics(p3.metricas)

        mlflow.sklearn.log_model(
            p3.modelo,
            name=modelo,
            input_example=p3.X.head(3),
            signature=infer_signature(p3.X, p3.etiquetas),
            serialization_format="skops",
        )

        generar_plots(run_id=run.info.run_id)

    return run.info.run_id


if __name__ == '__main__':
    app()
