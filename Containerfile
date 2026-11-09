# Imagen de la API del equipo 43: P1 (demanda por estación y día) y P2 (riesgo de que una estación
# se vacíe o se llene). Compatible con Podman y Docker.
#
# Antes de construir, exporta del Model Registry los modelos que se empaquetan (ver la sección de
# despliegue del README):
#     python -m mibici_mlops.api.modelo exportar                  # P1, en models/servicio/p1/
#     python -m mibici_mlops.api.modelo exportar --question p2    # P2, en models/servicio/p2/
# Construir y correr:
#     podman build -t mibici-api -f Containerfile .
#     podman run --rm -p 8000:8000 mibici-api
# (o lo mismo con `docker`).

# La misma versión de Python con la que se entrenó y guardó el modelo (python_env.yaml del modelo).
FROM python:3.13.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Las dependencias van primero, en su propia capa: si solo cambia el código o el modelo, no se
# vuelven a instalar.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# El paquete se usa desde /app (sin instalarlo): config.py toma /app como raíz del proyecto, así
# que los modelos quedan en /app/models/servicio/p1 y /app/models/servicio/p2, donde los busca
# cargar_modelo(). La API no arranca si falta alguno de los dos.
COPY mibici_mlops/ mibici_mlops/
COPY models/servicio/ models/servicio/

# No correr como root.
RUN useradd --create-home --uid 1000 api
USER api

EXPOSE 8000

# Docker la usa para marcar el contenedor como healthy; Podman solo con --format docker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/salud', timeout=4)"]

CMD ["uvicorn", "mibici_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
