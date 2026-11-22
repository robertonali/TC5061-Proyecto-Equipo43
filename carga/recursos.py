"""Muestrea `docker stats` (o `podman stats`) a un CSV mientras corre la prueba de carga.

    python carga/recursos.py reports/carga/recursos.csv [--intervalo 2] [--motor docker]

Corre hasta que se interrumpe (Ctrl+C o SIGTERM). Cada fila: momento, contenedor, % de CPU (100 =
un núcleo completo) y memoria en MiB.
"""

import argparse
import csv
from datetime import UTC, datetime
import json
import re
import signal
import subprocess
import time

UNIDADES = {"b": 1, "kb": 1e3, "kib": 1024, "mb": 1e6, "mib": 1024**2, "gb": 1e9, "gib": 1024**3}


def a_mib(texto: str) -> float:
    """'512.3MiB / 7.6GiB' -> 512.3"""
    m = re.match(r"\s*([\d.]+)\s*([a-zA-Z]+)", texto.split("/")[0])
    if not m:
        return float("nan")
    return float(m.group(1)) * UNIDADES[m.group(2).lower()] / 1024**2


def muestra(motor: str) -> list[dict]:
    salida = subprocess.run(
        [motor, "stats", "--no-stream", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    filas = []
    for linea in salida.splitlines():
        d = json.loads(linea)
        filas.append(
            {
                "contenedor": d.get("Name") or d.get("name"),
                "cpu_pct": float(str(d.get("CPUPerc", d.get("cpu_percent", "0"))).rstrip("%")),
                "mem_mib": a_mib(d.get("MemUsage", d.get("mem_usage", "0B"))),
            }
        )
    return filas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("salida")
    p.add_argument("--intervalo", type=float, default=2)
    p.add_argument("--motor", default="docker", choices=["docker", "podman"])
    args = p.parse_args()

    def detener(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, detener)
    with open(args.salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["momento", "contenedor", "cpu_pct", "mem_mib"])
        w.writeheader()
        try:
            while True:
                ahora = datetime.now(UTC).timestamp()
                for fila in muestra(args.motor):
                    w.writerow({"momento": round(ahora, 1), **fila})
                f.flush()
                time.sleep(args.intervalo)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
