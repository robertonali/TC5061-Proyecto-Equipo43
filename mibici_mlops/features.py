"""Variables y etiquetas para los modelos (paso interim → processed).

Lee `data/interim/viajes.parquet` y el catálogo de estaciones, y guarda en `data/processed/` una
tabla por pregunta. Ninguna variable usa información posterior al momento de la predicción.

P1 (demanda de viajes por estación y día): una fila por estación y día que se quiere predecir,
con el número de viajes que inician ahí, variables de calendario y de la estación, y el historial
de los días anteriores. Las reglas están en references/decisiones_p1.md.

P2 (riesgo de que una estación se vacíe o se llene): una fila por estación y ventana de 2 horas,
con la etiqueta, variables de calendario y la frecuencia de cada clase en las semanas anteriores.
Las reglas están en references/decisiones_p2.md y el análisis, en las secciones 12 y 13 de
notebooks/2.0-rc-p2-tamano-ventana.ipynb.

P3 (grupos de estaciones con patrones de uso parecidos): una fila por estación y conjunto
(entrenamiento o prueba), con la fracción de sus salidas y llegadas en cada día de la semana y
franja del día. Las reglas están en references/decisiones_p3.md y la exploración, en
notebooks/1.0-im-initial-eda.ipynb.

Uso:
    python -m mibici_mlops.features p1
    python -m mibici_mlops.features p2
    python -m mibici_mlops.features p3
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mibici_mlops.config import (
    ESTACIONES_PATH,
    FECHA_FIN,
    FECHA_INICIO,
    FECHA_INICIO_PRUEBA,
    FECHA_INICIO_VALIDACION,
    P1_FEATURES_PATH,
    P2_HORA_REINICIO,
    P2_INVENTARIO_INICIAL,
    P2_SEMANAS_HISTORIA,
    P2_VENTANA,
    P2_VENTANAS_PATH,
    P3_FRANJAS,
    P3_MIN_VIAJES,
    P3_PERFILES_PATH,
    VIAJES_PATH,
)
from mibici_mlops.modeling.registro import etiquetar_run_de_proyecto

app = typer.Typer()

# Clases de P2, en el orden en que ganan un empate en el baseline de frecuencia (decisión 7).
CLASES_P2 = ["se_vacia", "se_llena", "sin_riesgo"]

# Columnas que se guardan: la etiqueta y variables conocidas al momento de predecir. El inventario
# simulado (mínimo, máximo, final) no se guarda porque describe la misma ventana que se predice.
COLUMNAS_P2 = [
    "estacion",
    "ventana",
    "dia",
    "hora",
    "dia_semana",
    "feriado",
    "capacidad",
    "clase",
    "frec_se_vacia",
    "frec_se_llena",
    "frec_sin_riesgo",
]


def dia_de_simulacion(momentos: pd.Series) -> pd.Series:
    """Día de la simulación de cada momento: de las P2_HORA_REINICIO a la misma hora del siguiente.

    Por qué para P2: el inventario se reinicia a esa hora, cuando casi no hay viajes (decisión 4);
    lo que pasa de madrugada cuenta en el día anterior.
    """
    return (momentos - pd.Timedelta(hours=P2_HORA_REINICIO)).dt.normalize()


def construir_movimientos(viajes: pd.DataFrame, estaciones: np.ndarray) -> pd.DataFrame:
    """Convierte cada viaje en una salida (-1) en su origen y una llegada (+1) en su destino.

    Por qué para P2: MiBici no publica cuántas bicicletas hay en cada estación, así que el
    inventario se aproxima con estos movimientos. Se filtra por estación al final para que una
    llegada desde una estación fuera del catálogo sí cuente en su destino.
    """
    salidas = pd.DataFrame(
        {"estacion": viajes["origen_id"], "momento": viajes["inicio"], "cambio": -1}
    )
    llegadas = pd.DataFrame(
        {"estacion": viajes["destino_id"], "momento": viajes["fin"], "cambio": 1}
    )
    movimientos = pd.concat([salidas, llegadas], ignore_index=True)
    return movimientos[movimientos["estacion"].isin(estaciones)]


def simular_inventario(movimientos: pd.DataFrame, capacidad: pd.Series) -> pd.DataFrame:
    """Agrega el día de la simulación y el inventario estimado después de cada movimiento.

    Cada estación empieza el día con P2_INVENTARIO_INICIAL de su capacidad, sin redondear, y se le
    suman las llegadas y se le restan las salidas. Los movimientos del mismo segundo conservan su
    orden (primero las salidas) para que el resultado no dependa de la versión de pandas.

    Por qué para P2: el riesgo de una ventana depende de lo que pasó antes ese mismo día, no solo
    de los viajes de la ventana (decisión 4).
    """
    movimientos = movimientos.sort_values("momento", kind="stable", ignore_index=True)
    movimientos["dia"] = dia_de_simulacion(movimientos["momento"])
    movimientos["inventario"] = (
        P2_INVENTARIO_INICIAL * movimientos["estacion"].map(capacidad)
        + movimientos.groupby(["estacion", "dia"])["cambio"].cumsum()
    )
    return movimientos


def resumir_ventanas(movimientos: pd.DataFrame) -> pd.DataFrame:
    """Resume cada estación y ventana: inventario mínimo, máximo y final, y balance neto.

    Por qué para P2: la predicción es por ventana de 2 horas (decisión 2), y el mínimo y el máximo
    dicen si en algún momento de la ventana el inventario cruzó un umbral (decisión 5).
    """
    ventana = movimientos["momento"].dt.floor(P2_VENTANA).rename("ventana")
    return (
        movimientos.groupby(["estacion", ventana])
        .agg(
            minimo=("inventario", "min"),
            maximo=("inventario", "max"),
            final=("inventario", "last"),
            neto=("cambio", "sum"),
        )
        .reset_index()
    )


def completar_ventanas(por_ventana: pd.DataFrame, capacidad: pd.Series) -> pd.DataFrame:
    """Agrega las ventanas sin movimientos y completa el inventario de todas.

    Cada estación tiene todas sus ventanas desde su primer hasta su último día con movimientos. Una
    ventana sin viajes conserva el inventario con el que terminó la anterior (`ffill` dentro de
    cada estación y día), y si el día empieza sin viajes tiene el inventario inicial. El mínimo y
    el máximo también cuentan el inventario con el que empieza cada ventana. Se descarta el día
    incompleto anterior a FECHA_INICIO.

    Por qué para P2: si una estación quedó vacía y nadie llegó después, sigue vacía; sin estas
    ventanas se perdería ~20% de las ventanas "se vacía". Empezar en el primer día con viajes evita
    contar como "sin riesgo" los días en que la estación todavía no existía.
    """
    por_ventana = por_ventana.assign(dia=dia_de_simulacion(por_ventana["ventana"]))
    por_ventana = por_ventana[por_ventana["dia"] >= pd.Timestamp(FECHA_INICIO)]

    activa = por_ventana.groupby("estacion")["dia"].agg(["min", "max"])
    activa["desde"] = activa["min"] + pd.Timedelta(hours=P2_HORA_REINICIO)
    activa["hasta"] = (
        activa["max"] + pd.Timedelta(days=1, hours=P2_HORA_REINICIO) - pd.Timedelta(P2_VENTANA)
    )
    todas = pd.DataFrame(
        {"ventana": pd.date_range(activa["desde"].min(), activa["hasta"].max(), freq=P2_VENTANA)}
    )

    ventanas = activa[["desde", "hasta"]].reset_index().merge(todas, how="cross")
    ventanas = ventanas[ventanas["ventana"].between(ventanas["desde"], ventanas["hasta"])]
    ventanas = ventanas[["estacion", "ventana"]].merge(
        por_ventana.drop(columns="dia"), on=["estacion", "ventana"], how="left"
    )
    ventanas["dia"] = dia_de_simulacion(ventanas["ventana"])
    ventanas["capacidad"] = ventanas["estacion"].map(capacidad)

    ventanas["final"] = ventanas.groupby(["estacion", "dia"])["final"].ffill()
    ventanas["final"] = ventanas["final"].fillna(P2_INVENTARIO_INICIAL * ventanas["capacidad"])
    ventanas["neto"] = ventanas["neto"].fillna(0)
    ventanas["inicio"] = ventanas["final"] - ventanas["neto"]
    ventanas["minimo"] = ventanas[["minimo", "inicio"]].min(axis=1)
    ventanas["maximo"] = ventanas[["maximo", "inicio"]].max(axis=1)
    return ventanas


def etiquetar(ventanas: pd.DataFrame) -> pd.Series:
    """Clase de cada ventana: "se_vacia", "se_llena" o "sin_riesgo".

    "se_vacia" si el inventario llega a 0 o menos; "se_llena" si llega a la capacidad o más. Si
    pasan las dos cosas, gana el umbral más cercano al inventario con que empieza la ventana.

    Por qué para P2: es la variable objetivo (decisiones 1 y 5). Las dos clases de riesgo piden
    acciones opuestas al equipo de rebalanceo: llevar bicicletas o retirarlas.
    """
    se_vacia = ventanas["minimo"] <= 0
    se_llena = ventanas["maximo"] >= ventanas["capacidad"]
    empieza_abajo = ventanas["inicio"] <= ventanas["capacidad"] / 2

    clase = pd.Series("sin_riesgo", index=ventanas.index)
    clase[se_vacia] = "se_vacia"
    clase[se_llena] = "se_llena"
    clase[se_vacia & se_llena & empieza_abajo] = "se_vacia"
    return clase


def frecuencia_semanas_anteriores(
    ventanas: pd.DataFrame, semanas: int = P2_SEMANAS_HISTORIA
) -> pd.DataFrame:
    """Fracción de las semanas anteriores en que la misma estación y ventana tuvo cada clase.

    Compara con el mismo día de la semana y la misma hora: dentro de cada grupo (estación, día de
    la semana, hora), `shift(1)` es la misma ventana una semana antes. Supone que `ventanas` tiene
    todas las ventanas de cada estación en orden, como las deja `completar_ventanas`. Las semanas
    en que la estación todavía no existía cuentan como "sin riesgo".

    Por qué para P2: es el baseline de frecuencia (decisiones 3 y 7) y una variable para los
    modelos candidatos. Solo mira semanas que ya pasaron, así que no usa información posterior al
    momento de la predicción.
    """
    grupos = [ventanas["estacion"], ventanas["ventana"].dt.dayofweek, ventanas["ventana"].dt.hour]
    frecuencias = pd.DataFrame(index=ventanas.index)
    for clase in ["se_vacia", "se_llena"]:
        por_grupo = (ventanas["clase"] == clase).astype(float).groupby(grupos)
        anteriores = [por_grupo.shift(k).fillna(0) for k in range(1, semanas + 1)]
        frecuencias[f"frec_{clase}"] = sum(anteriores) / semanas
    frecuencias["frec_sin_riesgo"] = (
        1 - frecuencias["frec_se_vacia"] - frecuencias["frec_se_llena"]
    )
    return frecuencias


def variables_calendario(ventanas: pd.DataFrame) -> pd.DataFrame:
    """Hora en que empieza la ventana, día de la semana (0 = lunes) y si es día feriado.

    Se calculan con la fecha y hora de la ventana, igual que `frecuencia_semanas_anteriores`. Los
    feriados son los de `es_feriado`, la misma lista que usa P1.

    Por qué para P2: el riesgo se concentra en las horas pico de los días hábiles (decisión 2). En
    los feriados que caen entre semana, de 6 a 22 h solo el 6% de las ventanas está en riesgo,
    contra el 24% de un día hábil, y su perfil por hora se parece al de un fin de semana; la
    frecuencia de las semanas anteriores no lo sabe, porque compara con el mismo día de la semana.
    Las tres se conocen antes del momento de la predicción (4:00): no usan información del futuro.
    """
    momentos = ventanas["ventana"]
    return pd.DataFrame(
        {
            "hora": momentos.dt.hour,
            "dia_semana": momentos.dt.dayofweek,
            "feriado": es_feriado(pd.DatetimeIndex(momentos)),
        },
        index=ventanas.index,
    )


# Con un callback, cada pregunta es un subcomando (`features p2`) y P1 o P3 pueden agregar el suyo.
@app.callback()
def main(ctx: typer.Context):
    """Construye las tablas de variables y etiquetas en data/processed."""
    # Con `mlflow run . -e features -P question=p2`, etiqueta el run que creó MLflow.
    etiquetar_run_de_proyecto(str(ctx.invoked_subcommand).upper(), __file__, "features")


@app.command()
def p2(
    viajes_path: Path = VIAJES_PATH,
    estaciones_path: Path = ESTACIONES_PATH,
    output_path: Path = P2_VENTANAS_PATH,
    grupos: int = 8,
):
    """P2: etiqueta de riesgo, calendario y frecuencia de cada clase por estación y ventana."""
    if pd.Timedelta(hours=P2_HORA_REINICIO) % pd.Timedelta(P2_VENTANA):
        raise ValueError("P2_HORA_REINICIO debe coincidir con el inicio de una ventana")

    estaciones = pd.read_csv(estaciones_path)
    capacidad = estaciones[estaciones["dpcapacity"] > 0].set_index("id")["dpcapacity"]
    viajes = pd.read_parquet(
        viajes_path,
        columns=["inicio", "fin", "origen_id", "destino_id"],
        filters=[
            ("inicio", ">=", pd.Timestamp(FECHA_INICIO)),
            ("inicio", "<", pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)),
        ],
    )
    logger.info(f"{len(viajes):,} viajes de {FECHA_INICIO} a {FECHA_FIN}")

    # Por grupos de estaciones para usar menos memoria. El resultado es el mismo, porque el
    # inventario de cada estación solo depende de sus propios movimientos.
    partes = []
    for grupo in tqdm(np.array_split(np.sort(capacidad.index), grupos), desc="Grupos"):
        en_grupo = viajes["origen_id"].isin(grupo) | viajes["destino_id"].isin(grupo)
        movimientos = simular_inventario(construir_movimientos(viajes[en_grupo], grupo), capacidad)
        ventanas = completar_ventanas(resumir_ventanas(movimientos), capacidad)
        ventanas["clase"] = etiquetar(ventanas)
        ventanas = ventanas.join(frecuencia_semanas_anteriores(ventanas))
        ventanas = ventanas.join(variables_calendario(ventanas))
        partes.append(ventanas[COLUMNAS_P2])

    tabla = pd.concat(partes, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(output_path, index=False, compression="zstd")

    porcentajes = (tabla["clase"].value_counts(normalize=True) * 100).round(2).to_dict()
    logger.success(
        f"{len(tabla):,} ventanas de {tabla['estacion'].nunique()} estaciones guardadas en "
        f"{output_path}. Clases (%): {porcentajes}"
    )


# Días de la semana en el orden de `dt.dayofweek` (0 = lunes).
DIAS_SEMANA = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]


def columnas_perfil(prefijo: str) -> list[str]:
    """Nombres de las columnas del perfil de P3, p. ej. `sal_lun_manana` (decisión 2 de P3)."""
    return [f"{prefijo}_{dia}_{franja}" for dia in DIAS_SEMANA for franja in P3_FRANJAS]


def franja_del_dia(momentos: pd.Series) -> np.ndarray:
    """Número de la franja del día de cada momento: 0 = madrugada, 1 = mañana, etc.

    Las franjas empiezan a las horas de P3_FRANJAS. Por qué para P3: cuatro franjas bastan para
    distinguir los usos que importan a planeación urbana (ir al trabajo o la escuela en la
    mañana, regresar en la tarde, salir de noche) sin repartir los viajes de cada estación en
    tantas columnas que el perfil quede con ruido (decisión 2).
    """
    return np.searchsorted(list(P3_FRANJAS.values()), momentos.dt.hour, side="right") - 1


def perfil_de_uso(estacion: pd.Series, momentos: pd.Series, prefijo: str) -> pd.DataFrame:
    """Fracción de los viajes de cada estación en cada día de la semana y franja del día.

    Una fila por estación y 28 columnas (7 días × 4 franjas) que suman 1. Por qué para P3: al
    dividir entre el total de la estación se comparan sus patrones de uso, no su volumen; si no,
    los grupos solo separarían estaciones muy usadas de poco usadas (decisión 2).
    """
    codigo = momentos.dt.dayofweek.to_numpy() * len(P3_FRANJAS) + franja_del_dia(momentos)
    conteo = (
        pd.DataFrame({"estacion": estacion.to_numpy(), "codigo": codigo})
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=range(len(DIAS_SEMANA) * len(P3_FRANJAS)), fill_value=0)
    )
    conteo.columns = columnas_perfil(prefijo)
    return conteo.div(conteo.sum(axis=1), axis=0)


def perfiles_estaciones(
    viajes: pd.DataFrame,
    estaciones: np.ndarray,
    desde: str,
    hasta: str,
    min_viajes: int = P3_MIN_VIAJES,
) -> pd.DataFrame:
    """Perfil de salidas y llegadas de cada estación con los viajes que inician en [desde, hasta).

    Agrega el promedio de salidas y llegadas por día y el balance (llegadas - salidas) / total,
    que no se usan para agrupar sino para describir cada grupo. Deja fuera las estaciones que
    no están en el catálogo y las que tienen menos de `min_viajes` salidas o llegadas.

    Por qué para P3: las salidas dicen a qué hora la gente toma bicicletas en la estación y las
    llegadas a qué hora las deja; juntas distinguen, por ejemplo, una estación de vivienda (sale
    en la mañana, llega en la tarde) de una de trabajo (al revés). Con pocos viajes las
    fracciones son ruido (decisión 3). Recibir el periodo permite calcular el perfil solo con
    los datos de entrenamiento y, por separado, con los de prueba (decisión 4).
    """
    periodo = viajes[viajes["inicio"].between(desde, hasta, inclusive="left")]
    salidas = periodo["origen_id"].value_counts()
    llegadas = periodo["destino_id"].value_counts()
    dias = (pd.Timestamp(hasta) - pd.Timestamp(desde)).days

    perfil = perfil_de_uso(periodo["origen_id"], periodo["inicio"], "sal").join(
        perfil_de_uso(periodo["destino_id"], periodo["fin"], "lle"), how="inner"
    )
    perfil["salidas_dia"] = salidas / dias
    perfil["llegadas_dia"] = llegadas / dias
    perfil["balance"] = (llegadas - salidas) / (llegadas + salidas)

    suficientes = (salidas.reindex(perfil.index) >= min_viajes) & (
        llegadas.reindex(perfil.index) >= min_viajes
    )
    return perfil[perfil.index.isin(estaciones) & suficientes]


@app.command()
def p3(
    viajes_path: Path = VIAJES_PATH,
    estaciones_path: Path = ESTACIONES_PATH,
    output_path: Path = P3_PERFILES_PATH,
):
    """P3: perfil de uso de cada estación en los periodos de entrenamiento y de prueba."""
    estaciones = pd.read_csv(estaciones_path)["id"].to_numpy()
    viajes = pd.read_parquet(
        viajes_path,
        columns=["inicio", "fin", "origen_id", "destino_id"],
        filters=[
            ("inicio", ">=", pd.Timestamp(FECHA_INICIO)),
            ("inicio", "<", pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)),
        ],
    )
    logger.info(f"{len(viajes):,} viajes de {FECHA_INICIO} a {FECHA_FIN}")

    # La validación no se usa: P3 no ajusta hiperparámetros con ella (decisión 4).
    fin = (pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    periodos = {
        "entrenamiento": (FECHA_INICIO, FECHA_INICIO_VALIDACION),
        "validacion": (FECHA_INICIO_VALIDACION, FECHA_INICIO_PRUEBA),
        "prueba": (FECHA_INICIO_PRUEBA, fin),
    }
    partes = []
    for conjunto, (desde, hasta) in periodos.items():
        perfil = perfiles_estaciones(viajes, estaciones, desde, hasta)
        logger.info(f"{conjunto}: {len(perfil)} estaciones con perfil ({desde} a {hasta})")
        partes.append(perfil.reset_index().assign(conjunto=conjunto))

    tabla = pd.concat(partes, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(output_path, index=False, compression="zstd")
    logger.success(f"{len(tabla):,} perfiles guardados en {output_path}")


# Variables del modelo de P1. La variable objetivo es `viajes`.
COLUMNAS_P1 = [
    "dia_semana",
    "mes",
    "es_fin_de_semana",
    "es_feriado",
    "capacidad",
    "lat",
    "lon",
    "lag_7d",
    "media_7d",
]

DIAS_POR_SEMANA = 7


def dias_feriados(anios: range) -> set[pd.Timestamp]:
    """Días de descanso obligatorio de la Ley Federal del Trabajo (art. 74)."""

    def lunes(anio: int, mes: int, n: int) -> pd.Timestamp:
        primero = pd.Timestamp(anio, mes, 1)
        primer_lunes = primero + pd.Timedelta(days=(7 - primero.weekday()) % 7)
        return primer_lunes + pd.Timedelta(weeks=n - 1)

    feriados = set()
    for anio in anios:
        feriados |= {
            pd.Timestamp(anio, 1, 1),
            lunes(anio, 2, 1),  # Día de la Constitución
            lunes(anio, 3, 3),  # Natalicio de Benito Juárez
            pd.Timestamp(anio, 5, 1),
            pd.Timestamp(anio, 9, 16),
            lunes(anio, 11, 3),  # Día de la Revolución
            pd.Timestamp(anio, 12, 25),
        }
        if (anio - 1994) % 6 == 0:  # transmisión del Poder Ejecutivo Federal
            feriados.add(pd.Timestamp(anio, 10, 1))
    return feriados


def es_feriado(fechas: pd.DatetimeIndex) -> np.ndarray:
    """Indica, para cada fecha, si cae en un día de descanso obligatorio."""
    feriados = dias_feriados(range(fechas.year.min(), fechas.year.max() + 1))
    return fechas.normalize().isin(feriados)


def conteos_diarios(
    viajes: pd.DataFrame, estaciones_ids: pd.Index, dias: pd.DatetimeIndex
) -> np.ndarray:
    """Matriz (estaciones × días) con los viajes que inician en cada estación y día.

    La malla es completa: los días sin viajes quedan con 0. Los viajes de estaciones que no están
    en la nomenclatura se descartan y se reportan.

    Por qué para P1: los rezagos y promedios de `historial_demanda` cuentan columnas hacia atrás,
    así que cada columna debe ser exactamente un día.
    """
    t = ((viajes["inicio"].dt.floor("D") - dias[0]) // pd.Timedelta(days=1)).to_numpy()
    s = estaciones_ids.get_indexer(viajes["origen_id"].astype("int64"))
    conocida = s >= 0
    if not conocida.all():
        logger.warning(f"{(~conocida).sum():,} viajes con estación fuera de la nomenclatura")

    n_est, n_dias = len(estaciones_ids), len(dias)
    plano = s[conocida].astype(np.int64) * n_dias + t[conocida]
    matriz = np.bincount(plano, minlength=n_est * n_dias).reshape(n_est, n_dias)
    return matriz.astype(np.float32)


def historial_demanda(matriz: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula las variables de historial sin usar información del día que se predice.

    Para el día ``d``, el corte es el cierre del día ``d-1``. Todo rezago o ventana termina en
    ese día o antes. Donde no hay suficiente historial dentro de la malla, el valor es NaN.

    Parámetros:
        matriz: viajes por estación (filas) y día (columnas), sobre una malla diaria continua.
    """
    n_est, n_dias = matriz.shape

    # Mismo día de la semana anterior: d-7 (anterior al corte).
    lag_7d = np.full_like(matriz, np.nan)
    lag_7d[:, DIAS_POR_SEMANA:] = matriz[:, : n_dias - DIAS_POR_SEMANA]

    # Promedio de los 7 días anteriores (d-7 a d-1), con sumas acumuladas.
    suma = np.concatenate(
        [np.zeros((n_est, 1)), np.cumsum(matriz, axis=1, dtype=np.float64)], axis=1
    )
    media_7d = np.full_like(matriz, np.nan)
    d = np.arange(DIAS_POR_SEMANA, n_dias)
    media_7d[:, d] = (suma[:, d] - suma[:, d - DIAS_POR_SEMANA]) / DIAS_POR_SEMANA

    return {"lag_7d": lag_7d, "media_7d": media_7d}


def asignar_conjunto(fechas: pd.Series) -> pd.Categorical:
    """Etiqueta cada fila como entrenamiento, validación o prueba según su fecha.

    Usa la partición común de config.py (decisión 6 de P2), para que P1, P2 y P3 se evalúen en
    los mismos días.
    """
    conjunto = np.where(
        fechas < pd.Timestamp(FECHA_INICIO_VALIDACION),
        "entrenamiento",
        np.where(fechas < pd.Timestamp(FECHA_INICIO_PRUEBA), "validacion", "prueba"),
    )
    return pd.Categorical(conjunto, categories=["entrenamiento", "validacion", "prueba"])


def tabla_estacion_dia(
    matriz: np.ndarray, dias: pd.DatetimeIndex, estaciones: pd.DataFrame
) -> pd.DataFrame:
    """Arma la tabla estación-día con la variable objetivo y las variables de P1.

    Se conservan las filas con una semana completa de historial antes del corte y al menos un
    viaje en esa semana. Las estaciones nuevas o sin actividad reciente se excluyen: no tienen
    historial con qué predecir.
    """
    historial = historial_demanda(matriz)
    elegible = np.isfinite(historial["media_7d"]) & (historial["media_7d"] > 0)
    s_idx, t_idx = np.nonzero(elegible)

    fecha = dias[t_idx]
    tabla = pd.DataFrame(
        {
            "estacion_id": estaciones.index.to_numpy()[s_idx].astype("int32"),
            "fecha": fecha,
            "viajes": matriz[s_idx, t_idx].astype("int32"),
            "dia_semana": fecha.dayofweek.astype("int8"),
            "mes": fecha.month.astype("int8"),
            "es_fin_de_semana": (fecha.dayofweek >= 5).astype("int8"),
            "es_feriado": es_feriado(dias)[t_idx].astype("int8"),
            "capacidad": estaciones["dpcapacity"].to_numpy(dtype="float32")[s_idx],
            "lat": estaciones["latitude"].to_numpy(dtype="float32")[s_idx],
            "lon": estaciones["longitude"].to_numpy(dtype="float32")[s_idx],
        }
    )
    for nombre, valores in historial.items():
        tabla[nombre] = valores[s_idx, t_idx].astype("float32")
    tabla["conjunto"] = asignar_conjunto(tabla["fecha"])
    return tabla


@app.command()
def p1(
    viajes_path: Path = VIAJES_PATH,
    estaciones_path: Path = ESTACIONES_PATH,
    output_path: Path = P1_FEATURES_PATH,
):
    """P1: viajes por estación y día, con variables de calendario, de la estación y de historial.

    Solo se leen `inicio` y `origen_id`: los datos del usuario (`usuario_id`, `genero`,
    `anio_nacimiento`) no se cargan.
    """
    estaciones = pd.read_csv(estaciones_path).set_index("id").sort_index()
    logger.info(f"{len(estaciones)} estaciones en la nomenclatura")

    viajes = pd.read_parquet(
        viajes_path,
        columns=["inicio", "origen_id"],
        filters=[
            ("inicio", ">=", pd.Timestamp(FECHA_INICIO)),
            ("inicio", "<", pd.Timestamp(FECHA_FIN) + pd.Timedelta(days=1)),
        ],
    ).dropna()
    logger.info(f"{len(viajes):,} viajes de {FECHA_INICIO} a {FECHA_FIN}")

    dias = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    matriz = conteos_diarios(viajes, estaciones.index, dias)
    logger.info(f"Malla: {matriz.shape[0]} estaciones × {matriz.shape[1]:,} días")

    tabla = tabla_estacion_dia(matriz, dias, estaciones)
    excluidas = matriz.size - len(tabla)
    logger.info(
        f"{len(tabla):,} filas estación-día ({excluidas:,} excluidas por falta de historial "
        "o de actividad en la semana previa)"
    )
    for conjunto, grupo in tabla.groupby("conjunto", observed=True):
        logger.info(
            f"{conjunto}: {len(grupo):,} filas, {grupo['estacion_id'].nunique()} estaciones, "
            f"{grupo['fecha'].min():%Y-%m-%d} a {grupo['fecha'].max():%Y-%m-%d}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(output_path, compression="zstd", index=False)
    logger.success(f"Variables de P1 guardadas en {output_path}")


if __name__ == "__main__":
    app()
