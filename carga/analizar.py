"""Gráficas de la prueba de carga y cálculo FinOps del costo por inferencia.

    python carga/analizar.py reports/carga/w1 [reports/carga/w2 ...]

Lee de cada carpeta lo que deja carga/correr_carga.sh (locust_stats.csv,
locust_stats_history.csv y recursos.csv) y escribe en reports/carga/:

- <etiqueta>_desempeno.png  latencia p50/p95/p99, peticiones/s y errores contra el tiempo
- <etiqueta>_recursos.png   CPU y memoria de los contenedores contra el tiempo
- comparacion.png           peticiones/s y p95 por escalón de usuarios, una línea por corrida
- escalones.csv             las métricas de cada escalón de cada corrida
- finops.json y finops.md   el costo por inferencia y por millón de inferencias

Qué es una inferencia: una fila predicha. Una petición de 1 fila es 1 inferencia; una del proceso
diario es FILAS_LOTE inferencias; /salud y las inválidas (422) no son inferencias, pero sí consumen
la instancia, así que su costo se reparte entre las válidas.
"""

import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Objetivo de servicio (SLO) para decidir la capacidad sostenible de una corrida.
SLO_P95_MS = 500
SLO_ERRORES = 0.01

# Precios de lista on-demand de referencia (USD/hora, Linux). AWS us-east-1, GCP us-central1.
# Son referencias para el cálculo: verificar en la calculadora del proveedor antes de decidir.
INSTANCIAS = {
    "AWS t3.small (2 vCPU, 2 GiB)": 0.0208,
    "AWS t3.medium (2 vCPU, 4 GiB)": 0.0416,
    "AWS c6i.large (2 vCPU, 4 GiB)": 0.0850,
    "GCP e2-standard-2 (2 vCPU, 8 GiB)": 0.0670,
}
# AWS Fargate (us-east-1): se cobra por vCPU-hora y GB-hora de la tarea.
FARGATE_VCPU_H = 0.04048
FARGATE_GB_H = 0.004445
INSTANCIA_BASE = "AWS t3.medium (2 vCPU, 4 GiB)"  # la que imita el límite del compose

# Tráfico real estimado de MiBici para el escenario "demanda real" (ver finops.md).
ESTACIONES = 300
VENTANAS_P2 = 12  # ventanas de 2 h por día

SALIDA = Path("reports/carga")


def filas_por_peticion(nombre: str) -> int:
    if "inválida" in nombre or nombre == "salud":
        return 0
    m = re.search(r"lote (\d+)", nombre)
    return int(m.group(1)) if m else 1


def cargar(carpeta: Path) -> dict:
    hist = pd.read_csv(carpeta / "locust_stats_history.csv")
    hist = hist[hist["Name"] == "Aggregated"].copy()
    hist["t"] = hist["Timestamp"] - hist["Timestamp"].iloc[0]
    stats = pd.read_csv(carpeta / "locust_stats.csv")
    rec = pd.read_csv(carpeta / "recursos.csv")
    rec["t"] = rec["momento"] - hist["Timestamp"].iloc[0]
    rec["servicio"] = rec["contenedor"].str.extract(r"-(api|mlflow)-\d+$")[0]
    return {"etiqueta": carpeta.name, "hist": hist, "stats": stats, "rec": rec}


def escalones(c: dict) -> pd.DataFrame:
    """Métricas de cada escalón, sin los primeros 10 s (rampa) de cada uno."""
    h = c["hist"].copy()
    h["inicio"] = h.groupby("User Count")["t"].transform("min")
    estable = h[(h["t"] - h["inicio"] >= 10) & (h["User Count"] > 0)]
    filas = []
    for usuarios, g in estable.groupby("User Count"):
        t0, t1 = g["t"].min(), g["t"].max()
        api = c["rec"][(c["rec"]["servicio"] == "api") & c["rec"]["t"].between(t0, t1)]
        mlf = c["rec"][(c["rec"]["servicio"] == "mlflow") & c["rec"]["t"].between(t0, t1)]
        rps = g["Requests/s"].mean()
        filas.append(
            {
                "corrida": c["etiqueta"],
                "usuarios": int(usuarios),
                "rps": round(rps, 1),
                "p50_ms": round(g["50%"].median(), 0),
                "p95_ms": round(g["95%"].median(), 0),
                "p99_ms": round(g["99%"].median(), 0),
                "errores_pct": round(100 * g["Failures/s"].mean() / rps, 2) if rps else 0,
                "cpu_api_pct": round(api["cpu_pct"].mean(), 0),
                "mem_api_mib": round(api["mem_mib"].mean(), 0),
                "mem_mlflow_mib": round(mlf["mem_mib"].mean(), 0),
            }
        )
    return pd.DataFrame(filas)


def grafica_desempeno(c: dict) -> None:
    h = c["hist"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for col, color in (("50%", "#2a9d8f"), ("95%", "#e9c46a"), ("99%", "#e76f51")):
        a1.plot(h["t"], h[col], label=f"p{col[:-1]}", color=color)
    a1.axhline(SLO_P95_MS, ls="--", color="gray", lw=1, label=f"SLO p95 = {SLO_P95_MS} ms")
    a1.set_ylabel("Latencia (ms)")
    a1.set_yscale("log")
    a1.legend(loc="upper left")
    u = a1.twinx()
    u.plot(h["t"], h["User Count"], color="#264653", alpha=0.35, lw=2)
    u.set_ylabel("Usuarios", color="#264653")
    a2.plot(h["t"], h["Requests/s"], label="Peticiones/s", color="#264653")
    a2.plot(h["t"], h["Failures/s"], label="Fallas/s", color="#d62828")
    a2.set_ylabel("Peticiones por segundo")
    a2.set_xlabel("Segundos desde el inicio")
    a2.legend(loc="upper left")
    fig.suptitle(f"Prueba de carga — corrida {c['etiqueta']}")
    fig.tight_layout()
    fig.savefig(SALIDA / f"{c['etiqueta']}_desempeno.png", dpi=150)
    plt.close(fig)


def grafica_recursos(c: dict) -> None:
    r = c["rec"].dropna(subset=["servicio"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for servicio, g in r.groupby("servicio"):
        a1.plot(g["t"], g["cpu_pct"], label=servicio)
        a2.plot(g["t"], g["mem_mib"], label=servicio)
    a1.set_ylabel("CPU (%; 100 = 1 núcleo)")
    a1.legend(loc="upper left")
    a2.set_ylabel("Memoria (MiB)")
    a2.set_xlabel("Segundos desde el inicio")
    fig.suptitle(f"docker stats durante la carga — corrida {c['etiqueta']}")
    fig.tight_layout()
    fig.savefig(SALIDA / f"{c['etiqueta']}_recursos.png", dpi=150)
    plt.close(fig)


def grafica_comparacion(tabla: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for corrida, g in tabla.groupby("corrida"):
        a1.plot(g["usuarios"], g["rps"], marker="o", label=corrida)
        a2.plot(g["usuarios"], g["p95_ms"], marker="o", label=corrida)
    a1.set_xlabel("Usuarios concurrentes")
    a1.set_ylabel("Peticiones/s")
    a2.axhline(SLO_P95_MS, ls="--", color="gray", lw=1)
    a2.set_xlabel("Usuarios concurrentes")
    a2.set_ylabel("Latencia p95 (ms)")
    a2.set_yscale("log")
    for a in (a1, a2):
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle("Peticiones/s y latencia p95 por escalón, por configuración")
    fig.tight_layout()
    fig.savefig(SALIDA / "comparacion.png", dpi=150)
    plt.close(fig)


def finops(c: dict, tabla: pd.DataFrame) -> dict:
    s = c["stats"][c["stats"]["Name"] != "Aggregated"]
    peticiones = s["Request Count"].sum()
    inferencias = (s["Request Count"] * s["Name"].map(filas_por_peticion)).sum()
    inf_por_peticion = inferencias / peticiones

    t = tabla[tabla["corrida"] == c["etiqueta"]]
    ok = t[(t["p95_ms"] <= SLO_P95_MS) & (t["errores_pct"] <= 100 * SLO_ERRORES)]
    capacidad = ok.loc[ok["rps"].idxmax()] if len(ok) else t.loc[t["rps"].idxmax()]
    inf_s = capacidad["rps"] * inf_por_peticion
    inf_hora = inf_s * 3600

    mem_gb = (capacidad["mem_api_mib"] + capacidad["mem_mlflow_mib"]) / 1024
    costos = {
        nombre: {
            "usd_hora": precio,
            "usd_por_millon_inferencias": round(precio / inf_hora * 1e6, 4),
            "usd_por_peticion": precio / (capacidad["rps"] * 3600),
        }
        for nombre, precio in INSTANCIAS.items()
    }
    fargate_h = 2 * FARGATE_VCPU_H + 4 * FARGATE_GB_H  # tarea de 2 vCPU y 4 GB
    costos["AWS Fargate (2 vCPU, 4 GB)"] = {
        "usd_hora": round(fargate_h, 4),
        "usd_por_millon_inferencias": round(fargate_h / inf_hora * 1e6, 4),
        "usd_por_peticion": fargate_h / (capacidad["rps"] * 3600),
    }

    # Demanda real: el proceso diario predice P1 para cada estación y P2 para cada ventana.
    inf_dia_real = ESTACIONES * (1 + VENTANAS_P2)
    base = INSTANCIAS[INSTANCIA_BASE]
    return {
        "corrida": c["etiqueta"],
        "slo": {"p95_ms": SLO_P95_MS, "errores": SLO_ERRORES},
        "peticiones_totales": int(peticiones),
        "inferencias_totales": int(inferencias),
        "inferencias_por_peticion": round(inf_por_peticion, 2),
        "capacidad_sostenible": {
            "usuarios": int(capacidad["usuarios"]),
            "rps": float(capacidad["rps"]),
            "p95_ms": float(capacidad["p95_ms"]),
            "inferencias_por_s": round(inf_s, 0),
            "cpu_api_pct": float(capacidad["cpu_api_pct"]),
            "memoria_total_gb": round(mem_gb, 2),
        },
        "costos": costos,
        "demanda_real": {
            "inferencias_dia": inf_dia_real,
            "instancia_24x7_usd_mes": round(base * 24 * 30, 2),
            "usd_por_millon_inferencias_24x7": round(base * 24 / inf_dia_real * 1e6, 2),
            "utilizacion_pct": round(100 * inf_dia_real / (inf_hora * 24), 4),
            "fargate_10min_dia_usd_mes": round(fargate_h * (10 / 60) * 30, 2),
        },
    }


def markdown(resultados: list[dict], tabla: pd.DataFrame) -> str:
    lineas = [
        "# Resultados de carga y FinOps",
        "",
        "## Escalones",
        "",
        tabla.to_markdown(index=False),
        "",
    ]
    for r in resultados:
        cap = r["capacidad_sostenible"]
        lineas += [
            f"## Corrida {r['corrida']}",
            "",
            f"- Capacidad sostenible (p95 ≤ {SLO_P95_MS} ms, errores ≤ {100 * SLO_ERRORES:.0f} %): "
            f"{cap['usuarios']} usuarios, {cap['rps']} pet/s, p95 {cap['p95_ms']:.0f} ms, "
            f"CPU de la API {cap['cpu_api_pct']:.0f} %.",
            f"- {r['inferencias_por_peticion']} inferencias por petición en promedio → "
            f"{cap['inferencias_por_s']:.0f} inferencias/s.",
            "",
            "| Instancia | USD/h | USD por millón de inferencias | USD por petición |",
            "|---|---|---|---|",
        ]
        for nombre, c in r["costos"].items():
            lineas.append(
                f"| {nombre} | {c['usd_hora']:.4f} | {c['usd_por_millon_inferencias']:.4f} | "
                f"{c['usd_por_peticion']:.2e} |"
            )
        d = r["demanda_real"]
        lineas += [
            "",
            f"- Demanda real: {d['inferencias_dia']} inferencias/día. Una {INSTANCIA_BASE} 24x7 "
            f"cuesta {d['instancia_24x7_usd_mes']} USD/mes → {d['usd_por_millon_inferencias_24x7']} "
            f"USD por millón de inferencias, con {d['utilizacion_pct']} % de utilización.",
            f"- La misma carga como tarea de Fargate de 10 min al día: "
            f"{d['fargate_10min_dia_usd_mes']} USD/mes.",
            "",
        ]
    return "\n".join(lineas)


def main() -> None:
    corridas = [cargar(Path(p)) for p in sys.argv[1:]]
    tabla = pd.concat([escalones(c) for c in corridas], ignore_index=True)
    for c in corridas:
        grafica_desempeno(c)
        grafica_recursos(c)
    if len(corridas) > 1:
        grafica_comparacion(tabla)
    tabla.to_csv(SALIDA / "escalones.csv", index=False)
    resultados = [finops(c, tabla) for c in corridas]
    (SALIDA / "finops.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    (SALIDA / "finops.md").write_text(markdown(resultados, tabla))
    print(tabla.to_string(index=False))
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
