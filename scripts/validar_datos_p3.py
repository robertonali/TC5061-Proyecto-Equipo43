"""Validación de la tabla de perfiles de P3 con DeepChecks.

Se corre con el entorno aparte de DeepChecks (ver requirements-deepchecks.txt), no con .venv:

    .venv-deepchecks/bin/python scripts/validar_datos_p3.py

Corre dos suites y guarda un reporte HTML de cada una en reports/deepchecks/:
- Integridad (bloquea): tipos, valores faltantes, duplicados, columnas constantes, outliers y
  correlaciones inesperadas, sobre los perfiles de entrenamiento. Si alguna condición falla,
  el script termina con código 1, para que `make` o CI lo detecten.
- Drift (solo advierte): drift de cada variable y multivariado entre entrenamiento y prueba.
  Se reporta pero no detiene el pipeline: el drift entre periodos es esperado y se vigila con
  las métricas de estabilidad del modelo (decisión 7 de references/decisiones_p3.md).
"""

from itertools import combinations
from pathlib import Path
import sys
import warnings

from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import FeatureFeatureCorrelation
from deepchecks.tabular.suites import data_integrity, train_test_validation
import pandas as pd

warnings.filterwarnings("ignore")

PROJ_ROOT = Path(__file__).resolve().parents[1]
PERFILES_PATH = PROJ_ROOT / "data" / "processed" / "p3_perfiles.parquet"
REPORTES_DIR = PROJ_ROOT / "reports" / "deepchecks"

UMBRAL_CORRELACION = 0.9
DIAS_ENTRE_SEMANA = {"lun", "mar", "mie", "jue", "vie"}


def correlacion_esperada(a: str, b: str) -> bool:
    """Pares que se correlacionan por cómo se construye el perfil, no por un error en los datos.

    - La misma franja en dos días entre semana (p. ej. sal_lun_manana y sal_mar_manana): una
      estación que se usa para ir al trabajo lo hace igual de lunes a viernes.
    - salidas_dia y llegadas_dia: en una estación casi todas las bicicletas que salen regresan.
    """
    if {a, b} == {"salidas_dia", "llegadas_dia"}:
        return True
    partes_a, partes_b = a.split("_"), b.split("_")
    if len(partes_a) != 3 or len(partes_b) != 3:
        return False
    (sentido_a, dia_a, franja_a), (sentido_b, dia_b, franja_b) = partes_a, partes_b
    return sentido_a == sentido_b and franja_a == franja_b and {dia_a, dia_b} <= DIAS_ENTRE_SEMANA


def sin_correlaciones_inesperadas(correlaciones: pd.DataFrame) -> ConditionResult:
    """Falla si hay un par con correlación > UMBRAL que correlacion_esperada no explica."""
    altas = [
        (a, b)
        for a, b in combinations(correlaciones.columns, 2)
        if abs(correlaciones.loc[a, b]) > UMBRAL_CORRELACION
    ]
    inesperadas = [par for par in altas if not correlacion_esperada(*par)]
    if inesperadas:
        return ConditionResult(ConditionCategory.FAIL, f"Pares inesperados: {inesperadas}")
    return ConditionResult(
        ConditionCategory.PASS, f"{len(altas)} pares > {UMBRAL_CORRELACION}, todos esperados"
    )


def suite_integridad() -> Suite:
    """data_integrity, con la condición de correlación ajustada al perfil de uso de P3.

    La condición por defecto (ningún par > 0.9) siempre falla con el perfil de P3 por diseño
    (decisión 2): 48 pares en la primera corrida, todos esperados (decisión 7). En su lugar,
    solo falla con pares que no se explican por el diseño.
    """
    suite = data_integrity()
    for indice, chequeo in list(suite.checks.items()):
        if isinstance(chequeo, FeatureFeatureCorrelation):
            suite.remove(indice)
    suite.add(
        FeatureFeatureCorrelation().add_condition(
            f"Sin pares inesperados con correlación > {UMBRAL_CORRELACION}",
            sin_correlaciones_inesperadas,
        )
    )
    return suite


def como_advertencia(funcion):
    """Envuelve una condición de DeepChecks para que, si falla, se reporte como advertencia."""

    def condicion(*args, **kwargs):
        resultado = funcion(*args, **kwargs)
        if isinstance(resultado, bool):
            resultado = ConditionResult(
                ConditionCategory.PASS if resultado else ConditionCategory.FAIL
            )
        if resultado.category == ConditionCategory.FAIL:
            resultado.category = ConditionCategory.WARN
        return resultado

    return condicion


def suite_drift() -> Suite:
    """train_test_validation con sus condiciones como advertencias (decisión 7).

    El drift entre periodos es esperado en P3 y se vigila con las métricas de estabilidad del
    modelo, así que en el reporte HTML aparece como advertencia (!) y no como falla (✖).
    `_conditions` es interno de DeepChecks, pero la versión está fijada (0.19.1).
    """
    suite = train_test_validation()
    for chequeo in suite.checks.values():
        for condicion in chequeo._conditions.values():
            condicion.function = como_advertencia(condicion.function)
    return suite


def cargar_conjuntos() -> tuple[Dataset, Dataset]:
    """Datasets de DeepChecks de entrenamiento y prueba con las variables del perfil de uso.

    P3 es agrupamiento: no hay etiqueta. `estacion` no se pasa como índice porque las mismas
    estaciones aparecen a propósito en cada periodo (decisión 4 de P3); DeepChecks lo
    reportaría como fuga de índice entre entrenamiento y prueba.
    """
    perfiles = pd.read_parquet(PERFILES_PATH)
    variables = [c for c in perfiles.columns if c.startswith(("sal_", "lle_"))]
    variables += ["salidas_dia", "llegadas_dia", "balance"]
    conjuntos = {
        nombre: Dataset(
            perfiles.loc[perfiles["conjunto"] == nombre, variables],
            cat_features=[],
        )
        for nombre in ["entrenamiento", "prueba"]
    }
    return conjuntos["entrenamiento"], conjuntos["prueba"]


def main() -> int:
    entrenamiento, prueba = cargar_conjuntos()
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    # nombre del reporte: (resultado, si una condición que no se cumple detiene el pipeline)
    resultados = {
        "p3_integridad": (suite_integridad().run(entrenamiento), True),
        "p3_entrenamiento_vs_prueba": (suite_drift().run(entrenamiento, prueba), False),
    }
    bloqueado = False
    for nombre, (resultado, bloquea) in resultados.items():
        ruta = REPORTES_DIR / f"{nombre}.html"
        # HTML estático: el modo widget (por defecto) carga desde unpkg un html-manager de
        # ipywidgets 7 que no entiende el estado de ipywidgets 8, y el reporte se ve en blanco.
        # DeepChecks no sobrescribe: si el archivo existe, guarda una copia "nombre (1).html".
        ruta.unlink(missing_ok=True)
        resultado.save_as_html(str(ruta), as_widget=False)
        fallidas = resultado.get_not_passed_checks()
        nivel = "ERROR" if bloquea else "ADVERTENCIA"
        print(f"\n{nombre}: {len(fallidas)} chequeos con condiciones que no se cumplen")
        print(f"  reporte: {ruta.relative_to(PROJ_ROOT)}")
        for chequeo in fallidas:
            for condicion in chequeo.conditions_results:
                if not condicion.is_pass():
                    print(f"  [{nivel}] {chequeo.get_header()}: {condicion.name}")
                    print(f"      {condicion.details}")
        bloqueado = bloqueado or (bloquea and not resultado.passed())
    print("\nResultado:", "falla la integridad de los datos" if bloqueado else "integridad OK")
    return 1 if bloqueado else 0


if __name__ == "__main__":
    sys.exit(main())
