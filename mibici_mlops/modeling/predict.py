from loguru import logger
from pathlib import Path
from tqdm import tqdm

import pandas as pd

import mlflow
import typer

from mibici_mlops.config import MODELS_DIR, PROCESSED_DATA_DIR, P3_PERFILES_PATH, MLFLOW_TRACKING_URI

app = typer.Typer()


@app.command("p3")
def predict_p3(
    model_uri: str = typer.Option(..., "--model-uri"),
    input_path: Path = P3_PERFILES_PATH,
    output_path: Path = PROCESSED_DATA_DIR / "p3_predicciones.parquet",
    conjunto: str = "prueba",
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    perfiles = pd.read_parquet(input_path)
    filas = perfiles[perfiles["conjunto"] == conjunto].copy()

    columnas = [
        c for c in perfiles.columns
        if c.startswith(("sal_", "lle_"))
    ]
    X = filas[columnas].astype("float64")

    modelo = mlflow.pyfunc.load_model(model_uri)
    etiquetas = modelo.predict(X)

    resultado = filas[["estacion", "conjunto"]].copy()
    resultado["cluster"] = etiquetas

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(output_path, index=False, compression="zstd")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
