#!/usr/bin/env bash
# Corre la prueba de carga completa: levanta el sistema, mide CPU y memoria con `docker stats`
# mientras Locust genera la carga, y deja los resultados en reports/carga/<etiqueta>/.
#
#     bash carga/correr_carga.sh [etiqueta]        # p. ej. w1 (default) o w4
#
# Variables: las de docker-compose.yml (API_PORT, API_WORKERS, API_CPUS...) y las del locustfile
# (CARGA_ESCALONES, CARGA_SEG_ESCALON, CARGA_FILAS_LOTE). MOTOR=podman para usar Podman.
set -euo pipefail
cd "$(dirname "$0")/.."

ETIQUETA="${1:-w${API_WORKERS:-2}h${API_HILOS:-1}}"
MOTOR="${MOTOR:-docker}"
PUERTO="${API_PORT:-8000}"
SALIDA="reports/carga/${ETIQUETA}"
mkdir -p "$SALIDA"

echo "==> Levantando el sistema (API_WORKERS=${API_WORKERS:-2}, API_HILOS=${API_HILOS:-1}, API_CPUS=${API_CPUS:-2})"
$MOTOR compose up --build -d --wait

echo "==> Midiendo recursos en $SALIDA/recursos.csv"
python carga/recursos.py "$SALIDA/recursos.csv" --motor "$MOTOR" &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT
sleep 4  # unas muestras en reposo antes de la carga

echo "==> Locust contra http://localhost:$PUERTO"
locust -f carga/locustfile.py --host "http://localhost:$PUERTO" --headless \
    --csv "$SALIDA/locust" --csv-full-history --html "$SALIDA/locust.html" \
    --only-summary || true  # Locust sale con 1 si hubo fallas; igual queremos el análisis

sleep 4
kill $MONITOR 2>/dev/null || true
echo "==> Listo. Analizar con: python carga/analizar.py $SALIDA"
