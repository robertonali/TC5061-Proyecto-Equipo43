"""Prueba de carga de la API del equipo 43 (P1, P2 y P3) con Locust.

El escenario imita los dos tipos de cliente que tendría la API en operación:

- `ClienteInteractivo`: una app o tablero que consulta una estación a la vez. Es la mayoría del
  tráfico. Pide sobre todo P1 (demanda del día), luego P2 (riesgo en la próxima ventana) y, de vez
  en cuando, P3 (grupo de la estación) y /salud. Una fracción pequeña de sus peticiones es
  inválida a propósito: la API debe rechazarlas con 422 sin caerse.
- `ProcesoDiario`: el proceso por lotes que manda todas las estaciones juntas (~300 filas) a P1 y
  P2. Son pocas peticiones, pero cada una cuesta cientos de inferencias.

La carga sube en escalones (`CargaEscalonada`) para ver en qué punto la latencia se dispara.

Correr (con la API levantada con `docker compose up`):
    locust -f carga/locustfile.py --host http://localhost:8000 --headless \
        --csv reports/carga/locust --csv-full-history --html reports/carga/locust.html
O con la interfaz web: `locust -f carga/locustfile.py --host http://localhost:8000` y abrir
http://localhost:8089. El script carga/correr_carga.sh hace todo y además mide CPU y memoria.

Variables de entorno:
    CARGA_ESCALONES     usuarios de cada escalón, separados por comas (default 10,25,50,100,150)
    CARGA_SEG_ESCALON   segundos por escalón (default 60)
    CARGA_FILAS_LOTE    filas por petición del proceso diario (default 300)
"""

import os
import random

from locust import HttpUser, LoadTestShape, between, constant, tag, task

# Mismos rangos que valida la API (esquemas.py, p2.py y p3.py).
LAT = (20.64, 20.74)
LON = (-103.43, -103.30)
HORAS_P2 = list(range(0, 24, 2))
DIAS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]
FRANJAS = ["madrugada", "manana", "tarde", "noche"]
SABADO = 5

FILAS_LOTE = int(os.getenv("CARGA_FILAS_LOTE", "300"))


def fila_p1(estacion_id: int) -> dict:
    dia = random.randint(0, 6)
    media = round(random.uniform(0, 40), 2)
    return {
        "estacion_id": estacion_id,
        "dia_semana": dia,
        "mes": random.randint(1, 12),
        "es_fin_de_semana": int(dia >= SABADO),
        "es_feriado": int(random.random() < 0.03),
        "capacidad": random.choice([11, 15, 19, 23, 27, 31]),
        "lat": round(random.uniform(*LAT), 5),
        "lon": round(random.uniform(*LON), 5),
        "lag_7d": random.randint(0, 60),
        "media_7d": media,
    }


def fila_p2(estacion_id: int) -> dict:
    vacia = round(random.choice([0, 0.25, 0.5, 0.75, 1.0]), 2)
    llena = round(random.choice([x for x in (0, 0.25, 0.5, 0.75, 1.0) if x + vacia <= 1]), 2)
    return {
        "estacion_id": estacion_id,
        "hora": random.choice(HORAS_P2),
        "dia_semana": random.randint(0, 6),
        "feriado": random.random() < 0.03,
        "capacidad": random.choice([11, 15, 19, 23, 27, 31]),
        "frec_se_vacia": vacia,
        "frec_se_llena": llena,
    }


def fracciones(n: int) -> list[float]:
    """n fracciones no negativas que suman 1 (perfil de uso de P3)."""
    pesos = [random.expovariate(1) for _ in range(n)]
    total = sum(pesos)
    return [p / total for p in pesos]


def fila_p3(estacion_id: int) -> dict:
    columnas = [f"{d}_{f}" for d in DIAS for f in FRANJAS]
    fila = {"estacion_id": estacion_id}
    for prefijo in ("sal", "lle"):
        fila.update({f"{prefijo}_{c}": v for c, v in zip(columnas, fracciones(len(columnas)))})
    return fila


def estacion() -> int:
    return random.randint(1, 350)


class ClienteInteractivo(HttpUser):
    """Consulta una estación a la vez, con una pausa de lectura entre peticiones."""

    weight = 10
    wait_time = between(0.5, 2)

    @tag("p1")
    @task(6)
    def p1_una_estacion(self):
        self.client.post("/prediccion", json={"filas": [fila_p1(estacion())]}, name="P1 1 fila")

    @tag("p2")
    @task(3)
    def p2_una_ventana(self):
        self.client.post("/p2/prediccion", json={"filas": [fila_p2(estacion())]}, name="P2 1 fila")

    @tag("p3")
    @task(1)
    def p3_una_estacion(self):
        self.client.post("/p3/prediccion", json={"filas": [fila_p3(estacion())]}, name="P3 1 fila")

    @task(1)
    def salud(self):
        self.client.get("/salud", name="salud")

    @tag("invalida")
    @task(1)
    def peticion_invalida(self):
        """El fin de semana no coincide con el día: la API debe responder 422, no 500."""
        fila = fila_p1(estacion())
        fila["dia_semana"], fila["es_fin_de_semana"] = 1, 1
        with self.client.post(
            "/prediccion",
            json={"filas": [fila]},
            name="P1 inválida (espera 422)",
            catch_response=True,
        ) as r:
            if r.status_code == 422:
                r.success()
            else:
                r.failure(f"esperaba 422 y llegó {r.status_code}")


class ProcesoDiario(HttpUser):
    """El proceso por lotes: todas las estaciones en una sola petición."""

    fixed_count = 2
    wait_time = constant(5)

    @tag("lote")
    @task(2)
    def p1_lote(self):
        filas = [fila_p1(i) for i in range(1, FILAS_LOTE + 1)]
        self.client.post("/prediccion", json={"filas": filas}, name=f"P1 lote {FILAS_LOTE}")

    @tag("lote")
    @task(1)
    def p2_lote(self):
        filas = [fila_p2(i) for i in range(1, FILAS_LOTE + 1)]
        self.client.post("/p2/prediccion", json={"filas": filas}, name=f"P2 lote {FILAS_LOTE}")


class CargaEscalonada(LoadTestShape):
    """Sube los usuarios por escalones y termina al acabar el último."""

    escalones = [int(u) for u in os.getenv("CARGA_ESCALONES", "10,25,50,100,150").split(",")]
    segundos = int(os.getenv("CARGA_SEG_ESCALON", "60"))

    def tick(self):
        t = self.get_run_time()
        i = int(t // self.segundos)
        if i >= len(self.escalones):
            return None
        usuarios = self.escalones[i]
        return usuarios, max(usuarios, 10)
