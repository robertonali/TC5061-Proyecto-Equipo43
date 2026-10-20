from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Semilla común para todos los modelos
RANDOM_STATE = 67

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

# Archivos que usan varios módulos
VIAJES_PATH = INTERIM_DATA_DIR / "viajes.parquet"
ESTACIONES_PATH = RAW_DATA_DIR / "estaciones" / "nomenclatura_2026_09.csv"
P1_FEATURES_PATH = PROCESSED_DATA_DIR / "p1_estacion_dia.parquet"
P1_MODEL_PATH = MODELS_DIR / "p1.joblib"
P2_VENTANAS_PATH = PROCESSED_DATA_DIR / "p2_ventanas.parquet"
P3_CLUSTERS_PATH = PROCESSED_DATA_DIR / "p3_clusters.parquet"
P3_PERFILES_PATH = PROCESSED_DATA_DIR / "p3_perfiles.parquet"
P3_VIAJES_PATH = INTERIM_DATA_DIR / "p3_viajes.parquet"
P3_MODEL_PATH = MODELS_DIR / "p3.joblib"

# Página de datos abiertos de donde dataset.py descarga los CSV que falten en data/raw.
# Ver references/fuentes_datos.md.
URL_DATOS_ABIERTOS = "https://mibici.net/es/datos-abiertos/"

# Periodo de análisis: primer día con uso recuperado tras la pandemia.
# Ver references/decisiones.md, decisión 2.
FECHA_INICIO = "2022-03-01"
# Último día con datos. Fijarlo evita que los resultados cambien al descargar meses nuevos.
FECHA_FIN = "2026-09-30"

# Partición temporal por día: entrenamiento hasta antes de FECHA_INICIO_VALIDACION, validación
# hasta antes de FECHA_INICIO_PRUEBA y prueba hasta FECHA_FIN.
# Ver references/decisiones_p2.md, decisión 6.
FECHA_INICIO_VALIDACION = "2025-10-01"
FECHA_INICIO_PRUEBA = "2026-04-01"

# Registro de experimentos: un solo experimento de MLflow para P1, P2 y P3. Los runs se guardan en
# mlflow.db y los artefactos en mlruns/, en la raíz del proyecto, desde donde sea que se corra.
EXPERIMENTO_MLFLOW = "mibici-equipo43"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlruns"
SEMILLA = 43

# P2: etiqueta de riesgo y baseline de frecuencia. Ver references/decisiones_p2.md, decisiones 2
# (ventana), 3 (semanas de historia), 4 (reinicio e inventario inicial) y 5 (umbral).
# La hora de reinicio debe coincidir con el inicio de una ventana.
P2_VENTANA = "2h"
P2_HORA_REINICIO = 4
P2_INVENTARIO_INICIAL = 0.5
P2_SEMANAS_HISTORIA = 4

# P3: perfil de uso de cada estación y número de grupos. Ver references/decisiones_p3.md,
# decisiones 2 (franjas), 3 (mínimo de viajes) y 5 (número de grupos).
# Hora en que empieza cada franja del día; cada una dura hasta que empieza la siguiente.
P3_FRANJAS = {"madrugada": 0, "manana": 6, "tarde": 12, "noche": 18}
P3_MIN_VIAJES = 500
P3_GRUPOS = 3
P3_GRUPOS_BUSQUEDA = range(2, 9)

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
