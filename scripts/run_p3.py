"""Ejecuta la generación de perfiles, el entrenamiento y las gráficas de P3."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mibici_mlops.config import VIAJES_PATH

from mibici_mlops.dataset import main as preparar_datos
from mibici_mlops.features import p3 as generar_perfiles
from mibici_mlops.modeling.train_p3 import p3train as entrenar
from mibici_mlops.plots import p3 as generar_plots


def main():
    print("\n[PIPELINE] Preparando data/interim/viajes.parquet")
    if not VIAJES_PATH.exists():
        print("\n[PIPELINE] Preparando datos desde los CSV locales")
        preparar_datos(descargar_csv=False)
    else:
        print(f"\n[PIPELINE] Reutilizando {VIAJES_PATH}")

    print("\n[PIPELINE] Generando perfiles P3")
    generar_perfiles()

    print("\n[PIPELINE] Entrenando el modelo P3")
    run_id = entrenar()

    print(f"\n[PIPELINE] Run de MLflow: {run_id}")
    print("\n[PIPELINE] Generando mapa y tabla de perfiles")
    generar_plots(run_id=run_id)

    print("\n[PIPELINE] Flujo P3 completado")


if __name__ == "__main__":
    main()
