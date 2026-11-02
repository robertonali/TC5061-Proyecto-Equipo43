"""Validación de la tabla estación-día de P1 con DeepChecks.

Se corre con el entorno aparte de DeepChecks (ver requirements-deepchecks.txt y la sección de
pruebas del README), no con .venv:

    .venv-deepchecks\\Scripts\\python.exe scripts/validar_datos_p1.py

Lee data/processed/p1_estacion_dia.parquet (la salida de `features p1`), que ya trae la columna
`conjunto` con la partición común de config.py, y corre dos suites sobre una muestra aleatoria de
cada conjunto, con semilla fija:
- data_integrity, sobre entrenamiento: tipos, valores faltantes, duplicados, columnas constantes,
  valores atípicos, correlación entre variables y con la variable objetivo.
- train_test_validation, entrenamiento contra prueba: drift de cada variable y del objetivo,
  drift multivariado, categorías nuevas, y fugas por fecha o por filas repetidas.

Termina con código 1 si alguna condición no se cumple, para que `make` o CI lo detecten.

Guarda un reporte HTML de cada suite y la lista de hallazgos en reports/deepchecks/. Las
constantes (semilla, variables) se leen de config.py, features.py y train_p1.py para no
repetirlas; este entorno no puede importar mibici_mlops porque sus dependencias son otras.
"""

import ast
from itertools import combinations
from pathlib import Path
import sys
import warnings

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, SuiteResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import FeatureDrift, FeatureFeatureCorrelation, MultivariateDrift
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd

warnings.filterwarnings("ignore")

PROJ_ROOT = Path(__file__).resolve().parents[1]
TABLA_PATH = PROJ_ROOT / "data" / "processed" / "p1_estacion_dia.parquet"
REPORTES_DIR = PROJ_ROOT / "reports" / "deepchecks"
CONFIG = PROJ_ROOT / "mibici_mlops" / "config.py"
FEATURES = PROJ_ROOT / "mibici_mlops" / "features.py"
TRAIN_P1 = PROJ_ROOT / "mibici_mlops" / "modeling" / "train_p1.py"

# Filas de cada conjunto que se pasan a DeepChecks. Entrenamiento tiene ~434 mil; con la muestra
# las suites tardan menos de un minuto y los chequeos estadísticos no cambian.
MUESTRA = 200_000

# Hallazgos revisados y aceptados (ver suite_integridad y suite_entrenamiento_vs_prueba). Todo lo
# que no esté aquí sigue haciendo fallar la validación.
UMBRAL = 0.9
CORRELACIONES_ESPERADAS = {frozenset({"lag_7d", "media_7d"})}
UMBRAL_DRIFT = 0.2
DRIFT_ESPERADO = frozenset({"mes"})


def constante(archivo: Path, nombre: str):
    """Valor literal de `nombre = ...` en un módulo, sin importarlo."""
    for nodo in ast.parse(archivo.read_text(encoding="utf-8")).body:
        if isinstance(nodo, ast.Assign) and any(
            isinstance(objetivo, ast.Name) and objetivo.id == nombre for objetivo in nodo.targets
        ):
            return ast.literal_eval(nodo.value)
    raise KeyError(f"{nombre} no está en {archivo.name}")


SEMILLA = constante(CONFIG, "SEMILLA")
VARIABLES = constante(FEATURES, "COLUMNAS_P1")
OBJETIVO = constante(TRAIN_P1, "OBJETIVO")
# dia_semana y mes son las categóricas del modelo; los dos indicadores binarios también se
# declaran categóricos para que su drift se mida con Cramér's V y no con Kolmogorov-Smirnov.
CATEGORICAS = [*constante(TRAIN_P1, "CATEGORICAS"), "es_fin_de_semana", "es_feriado"]


def cargar_conjuntos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entrenamiento, validación y prueba, con la columna `conjunto` que escribe features p1."""
    tabla = pd.read_parquet(TABLA_PATH)
    conjunto = tabla["conjunto"].astype(str)
    return (
        tabla[conjunto == "entrenamiento"],
        tabla[conjunto == "validacion"],
        tabla[conjunto == "prueba"],
    )


def resumen(nombre: str, conjunto: pd.DataFrame) -> str:
    """Filas, estaciones, fechas y viajes por estación-día, sobre el conjunto completo."""
    return (
        f"{nombre}: {len(conjunto):,} filas estación-día de {conjunto['estacion_id'].nunique()} "
        f"estaciones, de {conjunto['fecha'].min():%Y-%m-%d} a {conjunto['fecha'].max():%Y-%m-%d}; "
        f"viajes por estación-día: media {conjunto[OBJETIVO].mean():.1f}, "
        f"mediana {conjunto[OBJETIVO].median():.0f}, máximo {conjunto[OBJETIVO].max():,}"
    )


def dataset(conjunto: pd.DataFrame, sin: frozenset[str] = frozenset()) -> Dataset:
    """Dataset de DeepChecks con una muestra del conjunto.

    `estacion_id` y `fecha` no son variables del modelo, pero se dejan en la tabla para que los
    duplicados se busquen con la clave real (estación y día). `estacion_id` no se declara como
    índice: las mismas estaciones aparecen a propósito en entrenamiento y prueba, y DeepChecks lo
    reportaría como fuga. `conjunto` se quita porque es constante dentro de cada conjunto. Las
    variables de `sin` se quitan de la tabla (para el drift multivariado sin `mes`).
    """
    muestra = conjunto.drop(columns=["conjunto", *sin]).sample(
        min(MUESTRA, len(conjunto)), random_state=SEMILLA
    )
    return Dataset(
        muestra,
        label=OBJETIVO,
        label_type="regression",
        features=[v for v in VARIABLES if v not in sin],
        cat_features=[v for v in CATEGORICAS if v not in sin],
        datetime_name="fecha",
    )


def pares_correlacionados(correlaciones: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (a, b)
        for a, b in combinations(correlaciones.columns, 2)
        if abs(correlaciones.loc[a, b]) > UMBRAL
    ]


def sin_correlaciones_inesperadas(correlaciones: pd.DataFrame) -> ConditionResult:
    """Falla si hay un par con correlación > UMBRAL que no esté en CORRELACIONES_ESPERADAS."""
    altas = pares_correlacionados(correlaciones)
    inesperadas = [par for par in altas if frozenset(par) not in CORRELACIONES_ESPERADAS]
    if inesperadas:
        return ConditionResult(ConditionCategory.FAIL, f"Pares inesperados: {inesperadas}")
    return ConditionResult(
        ConditionCategory.PASS, f"{len(altas)} pares > {UMBRAL}: {altas}, todos esperados"
    )


def sin_drift_salvo_mes(valor: dict) -> ConditionResult:
    """Falla si alguna variable, fuera de las de DRIFT_ESPERADO, tiene drift >= UMBRAL_DRIFT."""
    con_drift = {
        columna: round(info["Drift score"], 3)
        for columna, info in valor.items()
        if info["Drift score"] >= UMBRAL_DRIFT
    }
    inesperado = {c: v for c, v in con_drift.items() if c not in DRIFT_ESPERADO}
    if inesperado:
        return ConditionResult(ConditionCategory.FAIL, f"Drift inesperado: {inesperado}")
    return ConditionResult(
        ConditionCategory.PASS, f"Solo drift esperado por la partición: {con_drift}"
    )


def suite_integridad() -> Suite:
    """data_integrity, con la condición de correlación ajustada al diseño de P1.

    La condición por defecto (ningún par > 0.9) falla por `lag_7d` y `media_7d`: la media de los
    7 días anteriores incluye el día de hace 7 días, así que las dos variables se mueven juntas
    por construcción. El modelo de árboles lo tolera. Solo falla con pares que no se explican así.
    """
    suite = data_integrity()
    for indice, chequeo in list(suite.checks.items()):
        if isinstance(chequeo, FeatureFeatureCorrelation):
            suite.remove(indice)
    suite.add(
        FeatureFeatureCorrelation().add_condition(
            f"Sin pares inesperados con correlación > {UMBRAL}", sin_correlaciones_inesperadas
        )
    )
    return suite


def suite_entrenamiento_vs_prueba() -> Suite:
    """train_test_validation, con el drift de `mes` aceptado y documentado.

    La partición es por fechas (config.py): entrenamiento cubre todos los meses del año y prueba
    solo de abril a septiembre de 2026. `mes` cambia de distribución por diseño (Cramér's V ~0.53),
    no por un problema de datos. Por eso:
    - Feature Drift falla solo si otra variable pasa de UMBRAL_DRIFT.
    - Multivariate Drift con todas las variables se deja sin condición, para que su valor quede en
      el reporte y en hallazgos; la condición por defecto (< 0.25) se evalúa sin `mes` en
      suite_drift_sin_mes. Sin `mes` el drift baja de ~0.53 a ~0.21.
    """
    suite = train_test_validation()
    for indice, chequeo in list(suite.checks.items()):
        if isinstance(chequeo, (FeatureDrift, MultivariateDrift)):
            suite.remove(indice)
    suite.add(
        FeatureDrift().add_condition(
            f"Drift < {UMBRAL_DRIFT} en todas las variables salvo {sorted(DRIFT_ESPERADO)}",
            sin_drift_salvo_mes,
        )
    )
    suite.add(MultivariateDrift())
    return suite


def suite_drift_sin_mes() -> Suite:
    """Drift multivariado sin las variables de DRIFT_ESPERADO, con la condición por defecto."""
    return Suite(
        "Drift multivariado sin mes",
        MultivariateDrift().add_condition_overall_drift_value_less_than(0.25),
    )


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
    """Valores que conviene documentar aunque no fallen: drift de variables, objetivo y conjunto.

    Un drift por debajo del umbral también importa: dice cuánto cambió el periodo de prueba.
    """
    lineas = []
    for chequeo in resultado.results:
        if not isinstance(chequeo, CheckResult):
            continue
        encabezado, valor = chequeo.get_header(), chequeo.value
        if encabezado == "Feature Drift":
            puntajes = {col: round(info["Drift score"], 3) for col, info in valor.items()}
            lineas.append(f"  - Feature Drift (puntaje por variable): {puntajes}")
        elif encabezado == "Label Drift":
            lineas.append(f"  - Label Drift: {valor['Drift score']:.3f} ({valor['Method']})")
        elif encabezado == "Multivariate Drift":
            # Hay dos: el de suite_drift_sin_mes tiene condición; el de la suite completa, no.
            alcance = (
                "sin mes" if chequeo.conditions_results else "todas las variables, informativo"
            )
            lineas.append(
                f"  - Multivariate Drift ({alcance}): {valor['domain_classifier_drift_score']:.3f} "
                f"(AUC del clasificador de dominio {valor['domain_classifier_auc']:.3f})"
            )
        elif encabezado == "Feature-Feature Correlation":
            lineas.append(
                f"  - Feature-Feature Correlation: pares > {UMBRAL}: {pares_correlacionados(valor)}"
            )
    return lineas


def main() -> int:
    entrenamiento, validacion, prueba = cargar_conjuntos()
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    completa = pd.concat([entrenamiento, validacion, prueba])
    repetidas = completa.duplicated(["estacion_id", "fecha"]).sum()
    faltantes = completa.isna().sum().sum()
    negativos = (completa[OBJETIVO] < 0).sum()
    nuevas = set(prueba["estacion_id"]) - set(entrenamiento["estacion_id"])
    hallazgos = [
        "Tabla completa (data/processed/p1_estacion_dia.parquet):",
        resumen("entrenamiento", entrenamiento),
        resumen("validacion", validacion),
        resumen("prueba", prueba),
        (
            f"filas repetidas (misma estación y día): {repetidas:,}; faltantes: {faltantes:,}; "
            f"viajes negativos: {negativos:,}"
        ),
        (
            f"estaciones de prueba sin historial en entrenamiento: {len(nuevas)} "
            f"({prueba['estacion_id'].isin(nuevas).sum():,} filas de prueba)"
        ),
        f"Suites sobre una muestra de {MUESTRA:,} filas por conjunto (semilla {SEMILLA}).",
        (
            f"Hallazgos aceptados: correlación {sorted(map(sorted, CORRELACIONES_ESPERADAS))}, "
            f"drift de {sorted(DRIFT_ESPERADO)} (ver docstrings de las suites)."
        ),
    ]
    todo_bien = repetidas == 0 and faltantes == 0 and negativos == 0
    suites = {
        "p1_integridad": (suite_integridad(), [dataset(entrenamiento)]),
        "p1_entrenamiento_vs_prueba": (
            suite_entrenamiento_vs_prueba(),
            [dataset(entrenamiento), dataset(prueba)],
        ),
        "p1_drift_sin_mes": (
            suite_drift_sin_mes(),
            [dataset(entrenamiento, DRIFT_ESPERADO), dataset(prueba, DRIFT_ESPERADO)],
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
    (REPORTES_DIR / "p1_hallazgos.txt").write_text(texto + "\n", encoding="utf-8")
    return 0 if todo_bien else 1


if __name__ == "__main__":
    sys.exit(main())
