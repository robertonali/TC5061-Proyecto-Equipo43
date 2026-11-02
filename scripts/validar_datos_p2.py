"""Validación de la tabla de ventanas de P2 con DeepChecks.

Se corre con el entorno aparte de DeepChecks (ver requirements-deepchecks.txt y la sección de
pruebas del README), no con .venv:

    .venv-deepchecks\\Scripts\\python.exe scripts/validar_datos_p2.py

Lee data/processed/p2_ventanas.parquet (la salida de `features p2`), la parte por fechas igual que
train_p2.py y corre dos suites sobre una muestra aleatoria de cada conjunto, con semilla fija:
- data_integrity, sobre entrenamiento: tipos, valores faltantes, duplicados, columnas constantes,
  valores atípicos, correlación entre variables y con la etiqueta.
- train_test_validation, entrenamiento contra prueba: drift de cada variable y de la etiqueta,
  drift multivariado, categorías y etiquetas nuevas, y fugas por fecha o por filas repetidas.

Dos chequeos quedan como informativos, sin condición (ver `sin_condicion_de_filas_iguales`).
Termina con código 1 si alguna condición no se cumple, para que `make` o CI lo detecten.

Guarda un reporte HTML de cada suite y la lista de hallazgos en reports/deepchecks/. Las
constantes (fechas de la partición, semilla, variables) se leen de config.py y train_p2.py para no
repetirlas; este entorno no puede importar mibici_mlops porque sus dependencias son otras.
"""

import ast
from pathlib import Path
import sys
import warnings

from deepchecks.core import CheckResult, SuiteResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import ConflictingLabels, TrainTestSamplesMix
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd

warnings.filterwarnings("ignore")

PROJ_ROOT = Path(__file__).resolve().parents[1]
VENTANAS_PATH = PROJ_ROOT / "data" / "processed" / "p2_ventanas.parquet"
REPORTES_DIR = PROJ_ROOT / "reports" / "deepchecks"
CONFIG = PROJ_ROOT / "mibici_mlops" / "config.py"
TRAIN_P2 = PROJ_ROOT / "mibici_mlops" / "modeling" / "train_p2.py"

# Filas de cada conjunto que se pasan a DeepChecks. La tabla completa tiene ~7 millones; con la
# muestra las suites tardan minutos y los chequeos estadísticos no cambian.
MUESTRA = 300_000


def constante(archivo: Path, nombre: str):
    """Valor literal de `nombre = ...` en un módulo, sin importarlo."""
    for nodo in ast.parse(archivo.read_text(encoding="utf-8")).body:
        if isinstance(nodo, ast.Assign) and any(
            isinstance(objetivo, ast.Name) and objetivo.id == nombre for objetivo in nodo.targets
        ):
            return ast.literal_eval(nodo.value)
    raise KeyError(f"{nombre} no está en {archivo.name}")


FECHA_INICIO_VALIDACION = constante(CONFIG, "FECHA_INICIO_VALIDACION")
FECHA_INICIO_PRUEBA = constante(CONFIG, "FECHA_INICIO_PRUEBA")
SEMILLA = constante(CONFIG, "SEMILLA")
VARIABLES = constante(TRAIN_P2, "VARIABLES")
CATEGORICAS = constante(TRAIN_P2, "CATEGORICAS")


def cargar_conjuntos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entrenamiento, validación y prueba de la tabla completa, con la partición de train_p2."""
    ventanas = pd.read_parquet(VENTANAS_PATH)
    validacion = ventanas["dia"] >= pd.Timestamp(FECHA_INICIO_VALIDACION)
    prueba = ventanas["dia"] >= pd.Timestamp(FECHA_INICIO_PRUEBA)
    return ventanas[~validacion], ventanas[validacion & ~prueba], ventanas[prueba]


def resumen(nombre: str, conjunto: pd.DataFrame) -> str:
    """Filas, estaciones y proporción de cada clase, sobre el conjunto completo."""
    clases = (conjunto["clase"].value_counts(normalize=True) * 100).round(1)
    return (
        f"{nombre}: {len(conjunto):,} ventanas de {conjunto['estacion'].nunique()} estaciones, "
        f"de {conjunto['dia'].min():%Y-%m-%d} a {conjunto['dia'].max():%Y-%m-%d}; clases (%): "
        f"{clases.to_dict()}"
    )


def dataset(conjunto: pd.DataFrame) -> Dataset:
    """Dataset de DeepChecks con una muestra del conjunto.

    `estacion`, `dia` y `frec_sin_riesgo` no son variables del modelo, pero se dejan en la tabla
    para que los duplicados se busquen con la clave real (estación y ventana). `estacion` no se
    declara como índice: las mismas estaciones aparecen a propósito en entrenamiento y prueba, y
    DeepChecks lo reportaría como fuga.
    """
    muestra = conjunto.sample(min(MUESTRA, len(conjunto)), random_state=SEMILLA)
    return Dataset(
        muestra,
        label="clase",
        features=VARIABLES,
        cat_features=[*CATEGORICAS, "feriado"],
        datetime_name="ventana",
    )


def sin_condicion_de_filas_iguales(suite: Suite) -> Suite:
    """Deja como informativos los chequeos que buscan filas con las mismas variables.

    `ConflictingLabels` (mismas variables, distinta clase) y `TrainTestSamplesMix` (filas de
    prueba que ya estaban en entrenamiento) suponen que las variables identifican a cada fila. En
    P2 no: las 6 variables son gruesas (hora, día de la semana, feriado, capacidad y dos
    frecuencias con 5 niveles), así que miles de ventanas distintas comparten valores sin que sea
    un error ni una fuga. La identidad real de una fila es (estación, ventana), que no se repite,
    y las fugas por fecha las cubren los chequeos de fechas. Sus valores se reportan igual.
    """
    for chequeo in suite.checks.values():
        if isinstance(chequeo, (ConflictingLabels, TrainTestSamplesMix)):
            chequeo.clean_conditions()
    return suite


def correr(nombre: str, suite: Suite, *conjuntos: Dataset) -> SuiteResult:
    """Corre la suite y guarda su reporte HTML en reports/deepchecks/."""
    resultado = suite.run(*conjuntos)
    ruta = REPORTES_DIR / f"{nombre}.html"
    # HTML estático: el modo widget (por defecto) carga desde unpkg un html-manager de
    # ipywidgets 7 que no entiende el estado de ipywidgets 8, y el reporte se ve en blanco.
    # DeepChecks no sobrescribe: si el archivo existe, guarda una copia "nombre (1).html".
    ruta.unlink(missing_ok=True)
    resultado.save_as_html(str(ruta), as_widget=False)
    return resultado


def condiciones_no_cumplidas(resultado: SuiteResult) -> list[str]:
    return [
        f"  - {chequeo.get_header()}: {condicion.name} -> {condicion.details}"
        for chequeo in resultado.get_not_passed_checks()
        for condicion in chequeo.conditions_results
        if not condicion.is_pass()
    ]


def valores_informativos(resultado: SuiteResult) -> list[str]:
    """Valores que conviene documentar aunque no fallen: drift y filas con variables iguales.

    Un drift por debajo del umbral también importa: dice cuánto cambió el periodo de prueba.
    """
    lineas = []
    # `results` incluye los chequeos sin condición, que no cuentan como pasados ni fallidos.
    for chequeo in resultado.results:
        if not isinstance(chequeo, CheckResult):
            continue
        encabezado, valor = chequeo.get_header(), chequeo.value
        if encabezado == "Conflicting Labels":
            lineas.append(
                f"  - Conflicting Labels (informativo): {valor['percent_of_conflicting_samples']:.1%} "
                "de las ventanas comparten sus 6 variables con otra de distinta clase; esperado "
                "con variables gruesas, la clase no es función de las variables"
            )
        elif encabezado == "Train Test Samples Mix":
            lineas.append(
                f"  - Train Test Samples Mix (informativo): {valor['ratio']:.1%} de las ventanas "
                "de prueba tienen variables y clase iguales a alguna de entrenamiento; no es "
                "fuga, son ventanas distintas y las fechas no se traslapan"
            )
        elif encabezado == "Feature Drift":
            puntajes = {col: round(info["Drift score"], 3) for col, info in valor.items()}
            lineas.append(f"  - Feature Drift (puntaje por variable): {puntajes}")
        elif encabezado == "Label Drift":
            lineas.append(f"  - Label Drift: {valor['Drift score']:.3f} ({valor['Method']})")
        elif encabezado == "Multivariate Drift":
            lineas.append(
                f"  - Multivariate Drift: {valor['domain_classifier_drift_score']:.3f} "
                f"(AUC del clasificador de dominio {valor['domain_classifier_auc']:.3f})"
            )
    return lineas


def main() -> int:
    entrenamiento, validacion, prueba = cargar_conjuntos()
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    completa = pd.concat([entrenamiento, validacion, prueba])
    repetidas = completa.duplicated(["estacion", "ventana"]).sum()
    faltantes = completa.isna().sum().sum()
    hallazgos = [
        "Tabla completa (data/processed/p2_ventanas.parquet):",
        resumen("entrenamiento", entrenamiento),
        resumen("validacion", validacion),
        resumen("prueba", prueba),
        f"ventanas repetidas (misma estación y ventana): {repetidas:,}; faltantes: {faltantes:,}",
        f"Suites sobre una muestra de {MUESTRA:,} ventanas por conjunto (semilla {SEMILLA}).",
    ]
    todo_bien = True
    suites = {
        "p2_integridad": (
            sin_condicion_de_filas_iguales(data_integrity()),
            [dataset(entrenamiento)],
        ),
        "p2_entrenamiento_vs_prueba": (
            sin_condicion_de_filas_iguales(train_test_validation()),
            [dataset(entrenamiento), dataset(prueba)],
        ),
    }
    for nombre, (suite, conjuntos) in suites.items():
        resultado = correr(nombre, suite, *conjuntos)
        lineas = condiciones_no_cumplidas(resultado)
        hallazgos.append(
            f"\n{nombre}: {len(lineas)} condiciones que no se cumplen "
            f"(reports/deepchecks/{nombre}.html)"
        )
        hallazgos.extend(lineas + valores_informativos(resultado))
        todo_bien = todo_bien and resultado.passed()

    texto = "\n".join(hallazgos)
    print(texto)
    (REPORTES_DIR / "p2_hallazgos.txt").write_text(texto + "\n", encoding="utf-8")
    return 0 if todo_bien else 1


if __name__ == "__main__":
    sys.exit(main())
