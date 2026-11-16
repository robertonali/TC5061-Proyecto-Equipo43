"""A/B testing de P2: la versión alternativa contra el `champion` sobre las ventanas de `prueba`.

La versión alternativa es el run de `train_p2 --reentrenamiento` (etapa `retrain`): el mismo
modelo y los mismos hiperparámetros del `champion`, ajustado con entrenamiento y validación
(decisión 13 de references/decisiones_p2.md). Este script:

1. registra ese run como una nueva versión de P2_MODELO_REGISTRADO, con el alias `reentrenado`;
2. predice con las dos versiones las mismas ventanas de `prueba`, que ninguna usó para ajustarse,
   y compara la PR-AUC promedio de las clases de riesgo (la métrica con que se elige el modelo),
   el recall y la precisión;
3. mide qué tan confiable es la diferencia con bootstrap por días: repite la comparación con los
   días de `prueba` elegidos al azar con repetición y toma el intervalo donde cae el 95% de las
   diferencias;
4. aplica la regla de decisión (`gana`), fijada en config.py antes de correr el A/B;
5. deja el A/B como un run del experimento (`p2-ab-test`) y como etiquetas de las dos versiones;
6. con --promover, mueve el alias `champion` a la alternativa solo si gana.

Es un A/B fuera de línea: las dos versiones predicen las mismas ventanas históricas en lugar de
repartirse las estaciones. Se puede porque la etiqueta (si la estación se vació o se llenó) no
depende de lo que el modelo haya predicho, y comparar las mismas ventanas quita el ruido de
comparar días o estaciones distintos.

Uso (con el servidor de MLflow levantado):
    python -m mibici_mlops.modeling.ab_p2
    python -m mibici_mlops.modeling.ab_p2 --promover
"""

import json
from pathlib import Path

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import typer

from mibici_mlops.config import (
    EVIDENTLY_DIR,
    EXPERIMENTO_MLFLOW,
    P2_AB_CONFIANZA,
    P2_AB_MARGEN_PR_AUC,
    P2_AB_REPLICAS,
    P2_AB_TOLERANCIA_RECALL,
    P2_MODELO_REGISTRADO,
    P2_VENTANAS_PATH,
    SEMILLA,
)
from mibici_mlops.modeling.registrar_p2 import registrar_version
from mibici_mlops.modeling.registro import etiquetas_run, registrar_parametros, usar_experimento
from mibici_mlops.modeling.train_p2 import (
    CLASES_RIESGO,
    asignar_conjunto,
    calcular_metricas,
    pr_auc_por_mes,
    probabilidades_modelo,
)

app = typer.Typer()

CONJUNTO_EVALUACION = "prueba"
ALIAS_ALTERNATIVA = "reentrenado"
ALIAS_ANTERIOR = "anterior"
# Parámetros del run que definen el modelo. Si son iguales en las dos versiones, lo único que
# cambia entre ellas son los datos de entrenamiento.
HIPERPARAMETROS = [
    "modelo",
    "variables",
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "class_weight",
    "c",
]


def run_alternativo(cliente: MlflowClient) -> Run:
    """El run más reciente de reentrenamiento de P2 que terminó bien."""
    experimento = cliente.get_experiment_by_name(EXPERIMENTO_MLFLOW)
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string=(
            "tags.pregunta = 'P2' and tags.stage_of_work = 'retrain' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "No hay un run de reentrenamiento de P2. Corre `make reentrenar_p2` primero."
        )
    return runs[0]


def descripcion_del_run(run: Run) -> dict:
    """De qué run salió una versión: sus hiperparámetros y con qué ventanas se entrenó."""
    parametros = run.data.params
    # Los runs anteriores a la entrega 8 no registraron `entrenamiento_hasta`: entrenaron hasta el
    # día anterior al inicio de la validación.
    hasta = parametros.get("entrenamiento_hasta")
    if hasta is None:
        ultimo_dia = pd.Timestamp(parametros["fecha_inicio_validacion"]) - pd.Timedelta(days=1)
        hasta = f"{ultimo_dia:%Y-%m-%d}"
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "entrenamiento": {"ventanas": int(parametros["ventanas_entrenamiento"]), "hasta": hasta},
        "hiperparametros": {
            nombre: parametros[nombre] for nombre in HIPERPARAMETROS if nombre in parametros
        },
    }


def metricas_de_version(clase: pd.Series, probabilidades: pd.DataFrame) -> dict[str, float]:
    """Recall, precisión y PR-AUC de cada clase de riesgo y la PR-AUC promedio de una versión.

    Son las métricas del entrenamiento (train_p2.calcular_metricas), sin el prefijo del conjunto.
    """
    metricas = calcular_metricas(clase, probabilidades, "ab")
    return {
        clave.removeprefix("ab_"): float(valor)
        for clave, valor in metricas.items()
        if "proporcion" not in clave
    }


def diferencias_bootstrap(
    dia: pd.Series,
    clase: pd.Series,
    champion: pd.DataFrame,
    alternativa: pd.DataFrame,
    replicas: int = P2_AB_REPLICAS,
    semilla: int = SEMILLA,
) -> np.ndarray:
    """Diferencia de PR-AUC promedio (alternativa − champion) en cada réplica del bootstrap.

    Cada réplica elige al azar, con repetición, tantos días como tiene el conjunto y evalúa las
    dos versiones con las ventanas de esos días: una ventana pesa las veces que salió su día. Se
    remuestrean días completos y no ventanas sueltas porque las ventanas de un mismo día se
    parecen entre sí (clima, eventos); tratarlas como independientes haría ver la diferencia más
    segura de lo que es.
    """
    dias, dia_de_cada_ventana = np.unique(dia.to_numpy(), return_inverse=True)
    reales = {riesgo: (clase == riesgo).to_numpy() for riesgo in CLASES_RIESGO}

    def pr_auc_promedio(probabilidades: pd.DataFrame, pesos: np.ndarray) -> float:
        return np.mean(
            [
                average_precision_score(
                    reales[riesgo], probabilidades[riesgo].to_numpy(), sample_weight=pesos
                )
                for riesgo in CLASES_RIESGO
            ]
        )

    rng = np.random.default_rng(semilla)
    diferencias = np.empty(replicas)
    for replica in range(replicas):
        elegidos = rng.integers(0, len(dias), len(dias))
        pesos = np.bincount(elegidos, minlength=len(dias))[dia_de_cada_ventana]
        diferencias[replica] = pr_auc_promedio(alternativa, pesos) - pr_auc_promedio(
            champion, pesos
        )
    return diferencias


def intervalo_de_confianza(
    diferencias: np.ndarray, confianza: float = P2_AB_CONFIANZA
) -> tuple[float, float]:
    """Intervalo que contiene la fracción `confianza` central de las diferencias del bootstrap."""
    cola = (1 - confianza) / 2
    inferior, superior = np.quantile(diferencias, [cola, 1 - cola])
    return float(inferior), float(superior)


def gana(
    champion: dict,
    alternativa: dict,
    intervalo: tuple[float, float],
    margen: float = P2_AB_MARGEN_PR_AUC,
    tolerancia: float = P2_AB_TOLERANCIA_RECALL,
) -> tuple[bool, str]:
    """Regla de decisión: la alternativa reemplaza al champion solo si cumple tres condiciones.

    1. Mejora suficiente: la PR-AUC promedio sube al menos `margen`. Diferencias menores son del
       tamaño de las que hay entre combinaciones de hiperparámetros (decisión 9) y no justifican
       cambiar el modelo en producción.
    2. Mejora confiable: el intervalo de confianza de la diferencia queda arriba de 0, es decir,
       la alternativa gana aunque cambien los días con que se evalúa.
    3. Sin costo para el negocio: el recall de ninguna clase de riesgo baja más de `tolerancia`,
       porque el ML Canvas prioriza detectar las ventanas en riesgo.
    """
    mejora = alternativa["pr_auc_promedio"] - champion["pr_auc_promedio"]
    if mejora < margen:
        return False, f"la PR-AUC promedio cambia {mejora:+.4f}, menos que el margen de {margen}"
    if intervalo[0] <= 0:
        return False, (
            f"la mejora de {mejora:+.4f} no es confiable: su intervalo de confianza "
            f"({intervalo[0]:+.4f} a {intervalo[1]:+.4f}) incluye el 0"
        )
    for riesgo in CLASES_RIESGO:
        baja = champion[f"recall_{riesgo}"] - alternativa[f"recall_{riesgo}"]
        if baja > tolerancia:
            return False, (
                f"el recall de {riesgo} baja {baja:.4f}, más que la tolerancia de {tolerancia}"
            )
    return True, (
        f"la PR-AUC promedio mejora {mejora:+.4f} (intervalo de confianza de {intervalo[0]:+.4f} "
        f"a {intervalo[1]:+.4f}) y el recall de las clases de riesgo no baja más de {tolerancia}"
    )


def comparar_por_mes(
    evaluacion: pd.DataFrame, probabilidades: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """PR-AUC promedio de cada versión en cada mes del conjunto de evaluación y su diferencia."""
    tabla = pd.DataFrame(
        {
            nombre: pr_auc_por_mes({"test": evaluacion}, {"test": de_version}).set_index("mes")[
                "promedio"
            ]
            for nombre, de_version in probabilidades.items()
        }
    )
    tabla["diferencia"] = tabla["alternativa"] - tabla["champion"]
    tabla.index = tabla.index.strftime("%Y-%m")
    return tabla.round(4)


def cargar_version(version: str):
    return mlflow.sklearn.load_model(f"models:/{P2_MODELO_REGISTRADO}/{version}")


@app.command()
def main(
    ventanas_path: Path = P2_VENTANAS_PATH,
    salida_dir: Path = EVIDENTLY_DIR,
    replicas: int = typer.Option(P2_AB_REPLICAS, help="Réplicas del bootstrap por días."),
    promover: bool = typer.Option(
        False, help="Mueve el alias `champion` a la alternativa si gana el A/B."
    ),
):
    usar_experimento()
    cliente = MlflowClient()
    version_champion = cliente.get_model_version_by_alias(P2_MODELO_REGISTRADO, "champion")
    run = run_alternativo(cliente)
    versiones = {
        "champion": str(version_champion.version),
        "alternativa": str(registrar_version(cliente, run)),
    }
    if versiones["alternativa"] == versiones["champion"]:
        raise RuntimeError("La versión alternativa ya es el champion: no hay nada que comparar.")
    cliente.set_registered_model_alias(
        P2_MODELO_REGISTRADO, ALIAS_ALTERNATIVA, versiones["alternativa"]
    )
    logger.info(
        f"champion: versión {versiones['champion']}; alternativa: {versiones['alternativa']}"
    )
    runs = {"champion": cliente.get_run(version_champion.run_id), "alternativa": run}
    descripciones = {
        nombre: descripcion_del_run(de_version) for nombre, de_version in runs.items()
    }
    mismos_hiperparametros = (
        descripciones["champion"]["hiperparametros"]
        == descripciones["alternativa"]["hiperparametros"]
    )
    if not mismos_hiperparametros:
        logger.warning(
            "Las dos versiones no tienen los mismos hiperparámetros: el A/B no compara solo los "
            "datos de entrenamiento"
        )

    tabla = pd.read_parquet(ventanas_path)
    evaluacion = tabla[asignar_conjunto(tabla["dia"]) == CONJUNTO_EVALUACION]
    probabilidades = {
        nombre: probabilidades_modelo(cargar_version(version), evaluacion)
        for nombre, version in versiones.items()
    }
    metricas = {
        nombre: metricas_de_version(evaluacion["clase"], de_version)
        for nombre, de_version in probabilidades.items()
    }
    logger.info(f"Bootstrap por días: {replicas} réplicas (tarda unos minutos)")
    diferencias = diferencias_bootstrap(
        evaluacion["dia"],
        evaluacion["clase"],
        probabilidades["champion"],
        probabilidades["alternativa"],
        replicas,
    )
    intervalo = intervalo_de_confianza(diferencias)
    alternativa_gana, motivo = gana(metricas["champion"], metricas["alternativa"], intervalo)
    mejora = metricas["alternativa"]["pr_auc_promedio"] - metricas["champion"]["pr_auc_promedio"]
    por_mes = comparar_por_mes(evaluacion, probabilidades)

    resultado = {
        "conjunto_de_evaluacion": CONJUNTO_EVALUACION,
        "desde": f"{evaluacion['dia'].min():%Y-%m-%d}",
        "hasta": f"{evaluacion['dia'].max():%Y-%m-%d}",
        "dias": int(evaluacion["dia"].nunique()),
        "ventanas": len(evaluacion),
        **{
            nombre: {
                "version": versiones[nombre],
                **descripciones[nombre],
                **{clave: round(valor, 4) for clave, valor in metricas[nombre].items()},
            }
            for nombre in versiones
        },
        "mismos_hiperparametros": mismos_hiperparametros,
        "diferencia_pr_auc_promedio": round(mejora, 4),
        "intervalo_de_confianza": {
            "confianza": P2_AB_CONFIANZA,
            "inferior": round(intervalo[0], 4),
            "superior": round(intervalo[1], 4),
            "replicas": replicas,
            "semilla": SEMILLA,
            "replicas_en_que_gana_la_alternativa": round(float((diferencias > 0).mean()), 4),
        },
        "pr_auc_promedio_por_mes": por_mes.to_dict("index"),
        "meses_en_que_gana_la_alternativa": int((por_mes["diferencia"] > 0).sum()),
        "regla": {
            "margen_pr_auc": P2_AB_MARGEN_PR_AUC,
            "tolerancia_recall": P2_AB_TOLERANCIA_RECALL,
        },
        "gana_alternativa": alternativa_gana,
        "motivo": motivo,
    }
    salida_dir.mkdir(parents=True, exist_ok=True)
    archivo = salida_dir / "p2_ab_resultado.json"
    archivo.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        f"Métricas en {CONJUNTO_EVALUACION}:\n{pd.DataFrame(metricas).round(4).to_string()}"
    )
    logger.info(
        f"Diferencia de PR-AUC promedio: {mejora:+.4f}; intervalo de confianza de "
        f"{P2_AB_CONFIANZA:.0%}: {intervalo[0]:+.4f} a {intervalo[1]:+.4f}"
    )
    logger.info(f"PR-AUC promedio por mes:\n{por_mes.to_string()}")
    logger.info(f"Gana la alternativa: {alternativa_gana} ({motivo})")

    with mlflow.start_run(run_name="p2-ab-test") as ab:
        mlflow.set_tags(
            {
                **etiquetas_run("P2", __file__),
                "stage_of_work": "ab_test",
                "mlflow.runName": "p2-ab-test",
            }
        )
        registrar_parametros(
            {
                "conjunto_de_evaluacion": CONJUNTO_EVALUACION,
                "version_champion": versiones["champion"],
                "version_alternativa": versiones["alternativa"],
                "run_champion": runs["champion"].info.run_id,
                "run_alternativa": runs["alternativa"].info.run_id,
                "margen_pr_auc": P2_AB_MARGEN_PR_AUC,
                "tolerancia_recall": P2_AB_TOLERANCIA_RECALL,
                "confianza": P2_AB_CONFIANZA,
                "replicas": replicas,
                "semilla": SEMILLA,
                "promover": promover,
            }
        )
        for nombre, valores in metricas.items():
            mlflow.log_metrics({f"{nombre}_{clave}": valor for clave, valor in valores.items()})
        mlflow.log_metrics(
            {
                "diferencia_pr_auc_promedio": mejora,
                "intervalo_inferior": intervalo[0],
                "intervalo_superior": intervalo[1],
                "gana_alternativa": float(alternativa_gana),
            }
        )
        mlflow.log_artifact(str(archivo), artifact_path="ab")
        mlflow.set_tag("decision", motivo)

    for version in versiones.values():
        cliente.set_model_version_tag(P2_MODELO_REGISTRADO, version, "ab_test_run", ab.info.run_id)
    cliente.set_model_version_tag(
        P2_MODELO_REGISTRADO,
        versiones["alternativa"],
        "ab_test_gana",
        str(alternativa_gana).lower(),
    )

    if promover and alternativa_gana:
        cliente.set_registered_model_alias(
            P2_MODELO_REGISTRADO, ALIAS_ANTERIOR, versiones["champion"]
        )
        cliente.set_registered_model_alias(
            P2_MODELO_REGISTRADO, "champion", versiones["alternativa"]
        )
        logger.success(
            f"champion → versión {versiones['alternativa']} (la anterior queda como `anterior`)"
        )
    elif promover:
        logger.info(f"El champion sigue siendo la versión {versiones['champion']}")


if __name__ == "__main__":
    app()
