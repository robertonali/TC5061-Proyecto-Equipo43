# Resultados de carga y FinOps

## Escalones

| corrida   |   usuarios |   rps |   p50_ms |   p95_ms |   p99_ms |   errores_pct |   cpu_api_pct |   mem_api_mib |   mem_mlflow_mib |
|:----------|-----------:|------:|---------:|---------:|---------:|--------------:|--------------:|--------------:|-----------------:|
| base-w1   |         10 |   6.9 |       14 |       36 |       50 |          0.58 |            12 |           206 |             1638 |
| base-w1   |         50 |  37.6 |       12 |       48 |      110 |          0.05 |            64 |           209 |             1639 |
| base-w1   |        100 |  58.2 |      340 |     1300 |     1400 |          0.06 |           216 |           214 |             1639 |
| base-w1   |        200 |  54.9 |     2300 |     3600 |     3800 |          0.09 |           214 |           217 |             1639 |
| base-w1   |        300 |  54.6 |     4200 |     6200 |     6800 |          0.14 |           212 |           220 |             1639 |
| base-w2   |         10 |   6.8 |       14 |       34 |       48 |          0.79 |            11 |           423 |             1639 |
| base-w2   |         50 |  37.5 |       10 |       28 |       54 |          0    |            54 |           426 |             1639 |
| base-w2   |        100 |  45.7 |      700 |     2300 |     2600 |          0.1  |           210 |           434 |             1640 |
| base-w2   |        200 |  45.9 |     3350 |     6700 |     7200 |          0.11 |           213 |           439 |             1640 |
| base-w2   |        300 |  46.9 |     5200 |     8600 |    10000 |          0.09 |           212 |           441 |             1641 |
| opt-w1h2  |         10 |   6.6 |       18 |       39 |       53 |          0    |            15 |           204 |             1627 |
| opt-w1h2  |         50 |  37.6 |       30 |      140 |      290 |          0    |            94 |           207 |             1628 |
| opt-w1h2  |        100 |  57.5 |      340 |     1300 |     1500 |          0    |           216 |           212 |             1628 |
| opt-w1h2  |        200 |  59.2 |     2000 |     3200 |     3400 |          0    |           215 |           215 |             1628 |
| opt-w1h2  |        300 |  58.4 |     3900 |     5800 |     6300 |          0    |           213 |           218 |             1628 |
| opt-w2h1  |         10 |   6.6 |       15 |       36 |       78 |          0    |            11 |           426 |             1628 |
| opt-w2h1  |         50 |  38   |       10 |       30 |       58 |          0    |            41 |           428 |             1628 |
| opt-w2h1  |        100 |  76.6 |        9 |       32 |       58 |          0    |            52 |           432 |             1628 |
| opt-w2h1  |        200 | 125   |      205 |     1200 |     1700 |          0    |           191 |           439 |             1628 |
| opt-w2h1  |        300 | 112.7 |     1500 |     3000 |     3250 |          0    |           209 |           442 |             1629 |

## Corrida base-w1

- Capacidad sostenible (p95 ≤ 500 ms, errores ≤ 1 %): 50 usuarios, 37.6 pet/s, p95 48 ms, CPU de la API 64 %.
- 3.2 inferencias por petición en promedio → 120 inferencias/s.

| Instancia | USD/h | USD por millón de inferencias | USD por petición |
|---|---|---|---|
| AWS t3.small (2 vCPU, 2 GiB) | 0.0208 | 0.0480 | 1.54e-07 |
| AWS t3.medium (2 vCPU, 4 GiB) | 0.0416 | 0.0959 | 3.07e-07 |
| AWS c6i.large (2 vCPU, 4 GiB) | 0.0850 | 0.1960 | 6.28e-07 |
| GCP e2-standard-2 (2 vCPU, 8 GiB) | 0.0670 | 0.1545 | 4.95e-07 |
| AWS Fargate (2 vCPU, 4 GB) | 0.0987 | 0.2277 | 7.29e-07 |

- Demanda real: 3900 inferencias/día. Una AWS t3.medium (2 vCPU, 4 GiB) 24x7 cuesta 29.95 USD/mes → 256.0 USD por millón de inferencias, con 0.0375 % de utilización.
- La misma carga como tarea de Fargate de 10 min al día: 0.49 USD/mes.

## Corrida base-w2

- Capacidad sostenible (p95 ≤ 500 ms, errores ≤ 1 %): 50 usuarios, 37.5 pet/s, p95 28 ms, CPU de la API 54 %.
- 3.34 inferencias por petición en promedio → 125 inferencias/s.

| Instancia | USD/h | USD por millón de inferencias | USD por petición |
|---|---|---|---|
| AWS t3.small (2 vCPU, 2 GiB) | 0.0208 | 0.0462 | 1.54e-07 |
| AWS t3.medium (2 vCPU, 4 GiB) | 0.0416 | 0.0924 | 3.08e-07 |
| AWS c6i.large (2 vCPU, 4 GiB) | 0.0850 | 0.1888 | 6.30e-07 |
| GCP e2-standard-2 (2 vCPU, 8 GiB) | 0.0670 | 0.1488 | 4.96e-07 |
| AWS Fargate (2 vCPU, 4 GB) | 0.0987 | 0.2193 | 7.31e-07 |

- Demanda real: 3900 inferencias/día. Una AWS t3.medium (2 vCPU, 4 GiB) 24x7 cuesta 29.95 USD/mes → 256.0 USD por millón de inferencias, con 0.0361 % de utilización.
- La misma carga como tarea de Fargate de 10 min al día: 0.49 USD/mes.

## Corrida opt-w1h2

- Capacidad sostenible (p95 ≤ 500 ms, errores ≤ 1 %): 50 usuarios, 37.6 pet/s, p95 140 ms, CPU de la API 94 %.
- 3.06 inferencias por petición en promedio → 115 inferencias/s.

| Instancia | USD/h | USD por millón de inferencias | USD por petición |
|---|---|---|---|
| AWS t3.small (2 vCPU, 2 GiB) | 0.0208 | 0.0502 | 1.54e-07 |
| AWS t3.medium (2 vCPU, 4 GiB) | 0.0416 | 0.1004 | 3.07e-07 |
| AWS c6i.large (2 vCPU, 4 GiB) | 0.0850 | 0.2052 | 6.28e-07 |
| GCP e2-standard-2 (2 vCPU, 8 GiB) | 0.0670 | 0.1618 | 4.95e-07 |
| AWS Fargate (2 vCPU, 4 GB) | 0.0987 | 0.2384 | 7.29e-07 |

- Demanda real: 3900 inferencias/día. Una AWS t3.medium (2 vCPU, 4 GiB) 24x7 cuesta 29.95 USD/mes → 256.0 USD por millón de inferencias, con 0.0392 % de utilización.
- La misma carga como tarea de Fargate de 10 min al día: 0.49 USD/mes.

## Corrida opt-w2h1

- Capacidad sostenible (p95 ≤ 500 ms, errores ≤ 1 %): 100 usuarios, 76.6 pet/s, p95 32 ms, CPU de la API 52 %.
- 2.31 inferencias por petición en promedio → 177 inferencias/s.

| Instancia | USD/h | USD por millón de inferencias | USD por petición |
|---|---|---|---|
| AWS t3.small (2 vCPU, 2 GiB) | 0.0208 | 0.0327 | 7.54e-08 |
| AWS t3.medium (2 vCPU, 4 GiB) | 0.0416 | 0.0653 | 1.51e-07 |
| AWS c6i.large (2 vCPU, 4 GiB) | 0.0850 | 0.1335 | 3.08e-07 |
| GCP e2-standard-2 (2 vCPU, 8 GiB) | 0.0670 | 0.1052 | 2.43e-07 |
| AWS Fargate (2 vCPU, 4 GB) | 0.0987 | 0.1550 | 3.58e-07 |

- Demanda real: 3900 inferencias/día. Una AWS t3.medium (2 vCPU, 4 GiB) 24x7 cuesta 29.95 USD/mes → 256.0 USD por millón de inferencias, con 0.0255 % de utilización.
- La misma carga como tarea de Fargate de 10 min al día: 0.49 USD/mes.
