# Imagen de la API de MiBici: P1 (demanda por estación y día) y P3 (grupos de estaciones).
# Compatible con Podman y Docker.
#
# Antes de construir, exporta del Model Registry el modelo (o los modelos) que se empaquetan (ver
# la sección de despliegue del README):
#     python -m mibici_mlops.api.modelo exportar --pregunta p3
#     python -m mibici_mlops.api.modelo exportar --pregunta p1    # opcional
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
# que los modelos quedan en /app/models/servicio/p1 y /app/models/servicio/p3, donde los busca
# cargar_modelo(). Se copia la carpeta entera: la API sirve los modelos que estén exportados.
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
