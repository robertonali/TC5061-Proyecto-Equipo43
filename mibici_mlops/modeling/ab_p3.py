"""A/B testing de P3: la versión alternativa contra el `champion` sobre los perfiles de `prueba`.

La versión alternativa es el run de `train_p3 --reentrenamiento` (etapa `retrain`): el mismo
modelo y los mismos hiperparámetros del `champion`, ajustados con el perfil de toda la ventana de
entrenamiento y validación. Este script:
1. registra ese run como una nueva versión de P3_MODELO_REGISTRADO, con el alias `reentrenado`;
2. asigna grupos con las dos versiones a los mismos perfiles de `prueba`, que ninguna usó para
   ajustarse, y compara silhouette, Davies-Bouldin y Calinski-Harabasz;
3. deja el A/B como un run del experimento (`p3-ab-test`) y como etiquetas de la nueva versión;
4. con --promover, mueve el alias `champion` a la alternativa solo si gana (ver `gana`).

No hay etiquetas reales, así que se comparan métricas internas de los grupos, no el acierto. Los
números de grupo de las dos versiones no coinciden, por eso se miden las dos con las mismas
métricas y su acuerdo con el ARI entre sus asignaciones, no con una comparación de etiquetas.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.modeling.ab_p3
    python -m mibici_mlops.modeling.ab_p3 --promover
"""

import json
from pathlib import Path

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
import typer

from mibici_mlops.config import (
    EVIDENTLY_DIR,
    EXPERIMENTO_MLFLOW,
    P3_AB_MARGEN_SILHOUETTE,
    P3_MODELO_REGISTRADO,
    P3_PERFILES_PATH,
)
from mibici_mlops.features import COLUMNAS_P3
from mibici_mlops.modeling.registrar_p3 import registrar_version
from mibici_mlops.modeling.registro import etiquetas_run, registrar_parametros, usar_experimento

app = typer.Typer()

CONJUNTO_EVALUACION = "prueba"
ALIAS_ALTERNATIVA = "reentrenado"
ALIAS_ANTERIOR = "anterior"


def run_alternativo(cliente: MlflowClient) -> Run:
    """El run más reciente de reentrenamiento de P3 que terminó bien."""
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string=(
            "tags.pregunta = 'P3' and tags.stage_of_work = 'retrain' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "No hay un run de reentrenamiento de P3. Corre `make reentrenar_p3` primero."
        )
    return runs[0]


def metricas_de_grupos(perfil: pd.DataFrame, etiquetas) -> dict:
    """Métricas internas de los grupos en el espacio del perfil de uso."""
    return {
        "silhouette": float(silhouette_score(perfil, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(perfil, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(perfil, etiquetas)),
        "grupos": int(pd.Series(etiquetas).nunique()),
    }


def gana(
    champion: dict, alternativa: dict, margen: float = P3_AB_MARGEN_SILHOUETTE
) -> tuple[bool, str]:
    """Regla de decisión: la alternativa gana si mejora el silhouette en al menos `margen` y no
    empeora Davies-Bouldin. Dos métricas, porque una sola puede mejorar por azar; el margen,
    porque con unos cientos de estaciones diferencias de centésimas son ruido. Con distinto
    número de grupos las métricas no son comparables y la alternativa no gana.
    """
    if champion["grupos"] != alternativa["grupos"]:
        return False, "distinto número de grupos: las métricas no son comparables"
    mejora = alternativa["silhouette"] - champion["silhouette"]
    if mejora < margen:
        return False, f"el silhouette mejora {mejora:+.4f}, menos que el margen {margen}"
    if alternativa["davies_bouldin"] > champion["davies_bouldin"]:
        return False, "Davies-Bouldin empeora"
    return True, f"el silhouette mejora {mejora:+.4f} y Davies-Bouldin no empeora"


def cargar_version(version: str):
    return mlflow.sklearn.load_model(f"models:/{P3_MODELO_REGISTRADO}/{version}")


@app.command()
def main(
    perfiles_path: Path = P3_PERFILES_PATH,
    salida_dir: Path = EVIDENTLY_DIR,
    promover: bool = typer.Option(
        False, help="Mueve el alias `champion` a la alternativa si gana el A/B."
    ),
):
    usar_experimento()
    cliente = MlflowClient()
    version_champion = cliente.get_model_version_by_alias(P3_MODELO_REGISTRADO, "champion")
    run = run_alternativo(cliente)
    version_alternativa = registrar_version(cliente, run)
    cliente.set_registered_model_alias(
        P3_MODELO_REGISTRADO, ALIAS_ALTERNATIVA, version_alternativa
    )
    if version_alternativa == version_champion.version:
        raise RuntimeError("La versión alternativa ya es el champion: no hay nada que comparar.")
    logger.info(
        f"champion: versión {version_champion.version}; alternativa: {version_alternativa}"
    )

    perfiles = pd.read_parquet(perfiles_path)
    evaluacion = perfiles[perfiles["conjunto"].astype(str) == CONJUNTO_EVALUACION]
    perfil = evaluacion[COLUMNAS_P3].astype("float64")
    etiquetas = {
        "champion": cargar_version(version_champion.version).predict(perfil),
        "alternativa": cargar_version(version_alternativa).predict(perfil),
    }
    metricas = {nombre: metricas_de_grupos(perfil, e) for nombre, e in etiquetas.items()}
    alternativa_gana, motivo = gana(metricas["champion"], metricas["alternativa"])
    ari = float(adjusted_rand_score(etiquetas["champion"], etiquetas["alternativa"]))
    tamanos = {
        nombre: pd.Series(e).value_counts().sort_index().to_dict()
        for nombre, e in etiquetas.items()
    }
    resultado = {
        "conjunto_de_evaluacion": CONJUNTO_EVALUACION,
        "estaciones": len(evaluacion),
        "champion": {"version": version_champion.version, **metricas["champion"]},
        "alternativa": {
            "version": version_alternativa,
            "run_id": run.info.run_id,
            **metricas["alternativa"],
        },
        "ari_entre_versiones": round(ari, 4),
        "tamano_de_grupos": {
            n: {int(k): int(v) for k, v in t.items()} for n, t in tamanos.items()
        },
        "margen_silhouette": P3_AB_MARGEN_SILHOUETTE,
        "gana_alternativa": alternativa_gana,
        "motivo": motivo,
    }
    salida_dir.mkdir(parents=True, exist_ok=True)
    archivo = salida_dir / "p3_ab_resultado.json"
    archivo.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tabla = pd.DataFrame(metricas).T.round(4)
    logger.info(f"Métricas en {CONJUNTO_EVALUACION}:\n{tabla.to_string()}")
    logger.info(f"ARI entre versiones: {ari:.4f}")
    logger.info(f"Gana la alternativa: {alternativa_gana} ({motivo})")

    with mlflow.start_run(run_name="p3-ab-test") as ab:
        mlflow.set_tags(
            {
                **etiquetas_run("P3", __file__),
                "stage_of_work": "ab_test",
                "mlflow.runName": "p3-ab-test",
            }
        )
        registrar_parametros(
            {
                "conjunto_de_evaluacion": CONJUNTO_EVALUACION,
                "version_champion": version_champion.version,
                "version_alternativa": version_alternativa,
                "run_alternativa": run.info.run_id,
                "margen_silhouette": P3_AB_MARGEN_SILHOUETTE,
                "promover": promover,
            }
        )
        for nombre, valores in metricas.items():
            mlflow.log_metrics({f"{nombre}_{k}": v for k, v in valores.items()})
        mlflow.log_metrics(
            {"ari_entre_versiones": ari, "gana_alternativa": float(alternativa_gana)}
        )
        mlflow.log_artifact(str(archivo), artifact_path="ab")
        mlflow.set_tag("decision", motivo)

    for version in (version_champion.version, version_alternativa):
        cliente.set_model_version_tag(P3_MODELO_REGISTRADO, version, "ab_test_run", ab.info.run_id)
    cliente.set_model_version_tag(
        P3_MODELO_REGISTRADO, version_alternativa, "ab_test_gana", str(alternativa_gana).lower()
    )

    if promover and alternativa_gana:
        cliente.set_registered_model_alias(
            P3_MODELO_REGISTRADO, ALIAS_ANTERIOR, version_champion.version
        )
        cliente.set_registered_model_alias(P3_MODELO_REGISTRADO, "champion", version_alternativa)
        logger.success(
            f"champion → versión {version_alternativa} (la anterior queda como `anterior`)"
        )
    elif promover:
        logger.info(f"El champion sigue siendo la versión {version_champion.version}")


if __name__ == "__main__":
    app()
