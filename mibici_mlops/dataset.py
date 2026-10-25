"""Descarga, une y limpia los CSV mensuales de viajes de MiBici en un solo Parquet.

Paso raw → interim: descarga de la página de datos abiertos los CSV que falten en data/raw (si
se hizo `dvc pull`, no descarga nada), corrige las diferencias de formato entre archivos y los
errores de los datos, como filas duplicadas (ver references/fuentes_datos.md). No filtra filas
por criterios de análisis: esas decisiones, como el periodo a usar, van en features.py.

Al terminar escribe el manifiesto de la versión de los datos (data/interim/manifiesto.json): los
archivos usados con su md5 y sus filas. Los scripts de entrenamiento lo guardan en cada run. Con
`mlflow run . -e ingest`, la ingesta también queda como un run con el manifiesto.

Uso:
    python -m mibici_mlops.dataset
    python -m mibici_mlops.dataset --sin-descargar   # solo usa los CSV que ya están en data/raw
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from loguru import logger
import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import typer

from mibici_mlops.config import (
    ESTACIONES_PATH,
    FECHA_FIN,
    INTERIM_DATA_DIR,
    MANIFIESTO_PATH,
    RAW_DATA_DIR,
    URL_DATOS_ABIERTOS,
)
from mibici_mlops.modeling.registro import etiquetas_run, registrar_manifiesto, usar_experimento

app = typer.Typer()

# El servidor responde 406 a las peticiones sin User-Agent de navegador.
ENCABEZADOS = {"User-Agent": "Mozilla/5.0 (compatible; MiBici-MLOps/0.1)"}

# Abril y mayo de 2023 ya no están en la página: se usan las copias de Internet Archive
# (ver references/fuentes_datos.md, "Archivos recuperados de Internet Archive").
RESPALDOS = {
    "datos_abiertos_2023_04.csv": "https://web.archive.org/web/20261009131614id_/https://www.mibici.net/site/assets/files/46758/datos_abiertos_2023_04.csv",
    "datos_abiertos_2023_05.csv": "https://web.archive.org/web/20240617215841id_/https://www.mibici.net/site/assets/files/46759/datos_abiertos_2023_05.csv",
}

# Se renombra por posición porque el encabezado de la columna 4 viene dañado en varios archivos.
COLUMNAS = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio",
    "fin",
    "origen_id",
    "destino_id",
]

ESQUEMA = pa.schema(
    [
        ("viaje_id", pa.int64()),
        ("usuario_id", pa.int64()),
        ("genero", pa.string()),
        ("anio_nacimiento", pa.int16()),
        ("inicio", pa.timestamp("s")),
        ("fin", pa.timestamp("s")),
        ("origen_id", pa.int32()),
        ("destino_id", pa.int32()),
        ("origen", pa.string()),
        ("destino", pa.string()),
        ("origen_lon", pa.float64()),
        ("origen_lat", pa.float64()),
        ("destino_lon", pa.float64()),
        ("destino_lat", pa.float64()),
    ]
)

# Los nombres pueden traer sufijo (p. ej. datos_abiertos_2016_01-1.csv).
PATRON_ARCHIVO = re.compile(r"datos_abiertos_(\d{4})_(\d{2})")


def descargar(url: str, destino: Path) -> None:
    """Descarga `url` en `destino`. Escribe primero un .part para no dejar archivos a medias."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".part")
    with (
        urlopen(Request(url, headers=ENCABEZADOS), timeout=300) as respuesta,
        open(temporal, "wb") as archivo,
    ):
        shutil.copyfileobj(respuesta, archivo)
    temporal.rename(destino)


def enlaces_publicados(url: str = URL_DATOS_ABIERTOS) -> dict[str, str]:
    """Nombre → URL de cada CSV enlazado en la página de datos abiertos, más los respaldos."""
    with urlopen(Request(url, headers=ENCABEZADOS), timeout=60) as respuesta:
        html = respuesta.read().decode("utf-8", errors="replace")
    rutas = re.findall(r'href="([^"]+\.csv)"', html)
    enlaces = {Path(ruta).name: urljoin(url, ruta) for ruta in rutas}
    return {**RESPALDOS, **enlaces}


def descargar_faltantes(viajes_dir: Path, estaciones_path: Path) -> None:
    """Descarga la nomenclatura y los meses hasta FECHA_FIN que no estén en data/raw.

    Los meses posteriores a FECHA_FIN no se descargan, para que los datos no cambien cuando
    MiBici publique meses nuevos.
    """
    enlaces = enlaces_publicados()
    ultimo_mes = pd.Timestamp(FECHA_FIN).strftime("%Y_%m")
    pendientes = {}
    if not estaciones_path.exists():
        if estaciones_path.name not in enlaces:
            raise FileNotFoundError(
                f"{estaciones_path.name} ya no está publicado en {URL_DATOS_ABIERTOS}; "
                "obtenerlo con `dvc pull`"
            )
        pendientes[estaciones_path] = enlaces[estaciones_path.name]
    for nombre, url in enlaces.items():
        coincidencia = PATRON_ARCHIVO.search(nombre)
        if coincidencia is None or "_".join(coincidencia.groups()) > ultimo_mes:
            continue
        destino = viajes_dir / coincidencia.group(1) / nombre
        if not destino.exists():
            pendientes[destino] = url

    if not pendientes:
        logger.info("Todos los CSV ya están en data/raw; no se descarga nada")
        return
    logger.info(f"Descargando {len(pendientes)} archivos de {URL_DATOS_ABIERTOS}")
    for destino, url in tqdm(pendientes.items(), desc="Descargas"):
        descargar(url, destino)


def listar_archivos(viajes_dir: Path) -> list[Path]:
    """Regresa los CSV de viajes ordenados por año y mes."""
    archivos = []
    for ruta in viajes_dir.glob("*/*.csv"):
        coincidencia = PATRON_ARCHIVO.search(ruta.name)
        if coincidencia is None:
            logger.warning(f"Se ignora {ruta.name}: no sigue el patrón de nombre")
            continue
        archivos.append((coincidencia.groups(), ruta))
    return [ruta for _, ruta in sorted(archivos)]


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV mensual como texto, probando UTF-8 y luego Latin-1."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                ruta,
                encoding=encoding,
                header=0,
                names=COLUMNAS,
                dtype=str,
                na_values=["", "NULL", "NA"],
                keep_default_na=False,
            )
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {ruta}")


def parsear_fechas(columna: pd.Series) -> pd.Series:
    """Convierte fechas ISO (YYYY-MM-DD HH:MM:SS) o DD/MM/YYYY H:MM según el archivo."""
    muestra = columna.dropna().iloc[0]
    formato = "%d/%m/%Y %H:%M" if "/" in muestra else "%Y-%m-%d %H:%M:%S"
    return pd.to_datetime(columna, format=formato, errors="coerce")


def a_entero(columna: pd.Series, tipo: str) -> pd.Series:
    """Convierte a entero nullable; acepta flotantes como '1984.0'."""
    return pd.to_numeric(columna, errors="coerce").round().astype(tipo)

def a_float(columna: pd.Series) -> pd.Series:
    """Convierte a float nullable; acepta enteros como '1984'."""
    return pd.to_numeric(columna, errors="coerce").astype(float)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica tipos a un mes de viajes. Los valores inválidos quedan como nulos."""
    return pd.DataFrame(
        {
            "viaje_id": a_entero(df["viaje_id"], "Int64"),
            "usuario_id": a_entero(df["usuario_id"], "Int64"),
            "genero": df["genero"].str.strip().str.upper(),
            "anio_nacimiento": a_entero(df["anio_nacimiento"], "Int16"),
            "inicio": parsear_fechas(df["inicio"]),
            "fin": parsear_fechas(df["fin"]),
            "origen_id": a_entero(df["origen_id"], "Int32"),
            "destino_id": a_entero(df["destino_id"], "Int32"),
        }
    )


def reportar_nulos_nuevos(crudo: pd.DataFrame, limpio: pd.DataFrame, nombre: str) -> None:
    """Avisa cuando la conversión de tipos convirtió valores no vacíos en nulos."""
    perdidos = (limpio.isna() & crudo.notna()).sum()
    perdidos = perdidos[perdidos > 0]
    if not perdidos.empty:
        logger.warning(f"{nombre}: valores no convertibles -> nulo: {perdidos.to_dict()}")


def quitar_duplicados(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Elimina filas idénticas en todas las columnas (el mismo viaje escrito dos veces)."""
    sin_duplicados = df.drop_duplicates()
    eliminadas = len(df) - len(sin_duplicados)
    if eliminadas:
        logger.warning(f"{nombre}: {eliminadas:,} filas duplicadas eliminadas")
    return sin_duplicados

def unir_estaciones(df: pd.DataFrame, estdf: pd.DataFrame) -> pd.DataFrame:
    """Une las estaciones a los viajes."""
    con_est = df.copy()
    con_est["origen"] = con_est.origen_id.map(estdf.name)
    con_est["origen_lon"] = a_float(con_est.origen_id.map(estdf.longitude))
    con_est["origen_lat"] = a_float(con_est.origen_id.map(estdf.latitude))
    con_est["destino"] = con_est.destino_id.map(estdf.name)
    con_est["destino_lon"] = a_float(con_est.destino_id.map(estdf.longitude))
    con_est["destino_lat"] = a_float(con_est.destino_id.map(estdf.latitude))
    return con_est


def md5_archivo(ruta: Path) -> str:
    """md5 de un archivo, leído por bloques para no cargarlo completo en memoria."""
    md5 = hashlib.md5()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            md5.update(bloque)
    return md5.hexdigest()


def escribir_manifiesto(
    archivos: list[dict], estaciones_path: Path, total: int, ruta: Path
) -> dict:
    """Guarda el manifiesto de la versión de los datos y lo regresa.

    Lleva cada CSV con su md5 y sus filas después de quitar duplicados, la nomenclatura de
    estaciones con su md5 y el total de viajes. `version` es el md5 de todo lo anterior: cambia si
    cambia cualquier archivo o la limpieza. No lleva la fecha en que se generó, para que los mismos
    datos den siempre el mismo manifiesto.
    """
    manifiesto = {
        "fecha_fin": FECHA_FIN,
        "viajes": total,
        "estaciones": {"archivo": estaciones_path.name, "md5": md5_archivo(estaciones_path)},
        "archivos": archivos,
    }
    contenido = json.dumps(manifiesto, sort_keys=True).encode()
    manifiesto["version"] = hashlib.md5(contenido).hexdigest()
    ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifiesto


def registrar_ingesta(manifiesto_path: Path) -> None:
    """Agrega el manifiesto al run que creó `mlflow run . -e ingest`.

    `mlflow run` crea el run antes de llamar a este script y pasa su id en MLFLOW_RUN_ID;
    `mlflow.start_run()` lo retoma. Así la versión de los datos también queda en el experimento.
    """
    usar_experimento()
    etiquetas = {**etiquetas_run("datos", __file__), "stage_of_work": "ingest"}
    with mlflow.start_run(run_name="ingesta-datos", tags=etiquetas):
        mlflow.set_tag("mlflow.runName", "ingesta-datos")
        registrar_manifiesto(manifiesto_path)


@app.command()
def main(
    estaciones_path: Path = ESTACIONES_PATH,
    viajes_dir: Path = RAW_DATA_DIR / "viajes",
    output_path: Path = INTERIM_DATA_DIR / "viajes.parquet",
    manifiesto_path: Path = MANIFIESTO_PATH,
    descargar_csv: bool = typer.Option(True, "--descargar/--sin-descargar"),
):
    if descargar_csv:
        descargar_faltantes(viajes_dir, estaciones_path)

    estaciones = pd.read_csv(estaciones_path)
    estaciones.set_index("id", inplace=True)

    archivos = listar_archivos(viajes_dir)
    logger.info(f"Uniendo {len(archivos)} archivos de {viajes_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    usados = []
    # Se escribe mes por mes para no tener los ~39 M de filas en memoria a la vez.
    with pq.ParquetWriter(output_path, ESQUEMA, compression="zstd") as writer:
        for ruta in tqdm(archivos, desc="Meses"):
            crudo = leer_csv(ruta)
            limpio = limpiar(crudo)
            limpio = unir_estaciones(limpio, estaciones)
            reportar_nulos_nuevos(crudo, limpio, ruta.name)
            limpio = quitar_duplicados(limpio, ruta.name)
            writer.write_table(pa.Table.from_pandas(limpio, schema=ESQUEMA, preserve_index=False))
            total += len(limpio)
            usados.append({"archivo": ruta.name, "md5": md5_archivo(ruta), "filas": len(limpio)})

    logger.success(f"{total:,} viajes guardados en {output_path}")
    manifiesto = escribir_manifiesto(usados, estaciones_path, total, manifiesto_path)
    logger.success(f"Versión de los datos {manifiesto['version']} en {manifiesto_path}")
    if os.environ.get("MLFLOW_RUN_ID"):
        registrar_ingesta(manifiesto_path)


if __name__ == "__main__":
    app()
