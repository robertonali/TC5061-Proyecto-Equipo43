#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MiBici-MLOps
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mibici_mlops/dataset.py


## Reporte de drift y calidad de datos de P1 con Evidently (reports/evidently/)
.PHONY: monitoreo
monitoreo:
	$(PYTHON_INTERPRETER) -m mibici_mlops.monitoreo p1

## Drift y calidad de datos de P3 con Evidently, y registro en MLflow (requiere el servidor de MLflow)
.PHONY: monitoreo_p3
monitoreo_p3:
	$(PYTHON_INTERPRETER) -m mibici_mlops.monitoreo p3

## Versión alternativa de P3: perfiles ampliados, reentrenamiento y registro (requiere MLflow)
MODELO_P3 ?= kmeans_k3
.PHONY: reentrenar_p3
reentrenar_p3:
	$(PYTHON_INTERPRETER) -m mibici_mlops.features p3-ampliado
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.train_p3 --modelo $(MODELO_P3) --reentrenamiento \
		--input-path data/processed/p3_perfiles_ampliado.parquet \
		--output-path data/processed/p3_clusters_ampliado.parquet \
		--model-path models/p3_ampliado.joblib

## A/B testing de P3: registra la versión alternativa y la compara con el champion sobre prueba
.PHONY: ab_p3
ab_p3:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.ab_p3

## Drift, calidad de datos y desempeño del champion de P2 con Evidently (requiere el servidor de MLflow)
.PHONY: monitoreo_p2
monitoreo_p2:
	$(PYTHON_INTERPRETER) -m mibici_mlops.monitoreo p2

## Versión alternativa de P2: el modelo del champion reentrenado con entrenamiento y validación (requiere MLflow)
MODELO_P2 ?= --modelo hgb --learning-rate 0.05
.PHONY: reentrenar_p2
reentrenar_p2:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.train_p2 $(MODELO_P2) --reentrenamiento

## A/B testing de P2: registra la versión alternativa y la compara con el champion sobre prueba
.PHONY: ab_p2
ab_p2:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.ab_p2


## Pruebas unitarias con PyTest
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest

# Vulnerabilidades aceptadas, sin versión corregida que el proyecto pueda usar (ver README,
# "Chequeos de seguridad"): diskcache (dependencia de DVC) y setuptools (fijado por DeepChecks).
VULNS_ACEPTADAS_PROYECTO = --ignore-vuln PYSEC-2026-2447
VULNS_ACEPTADAS_DEEPCHECKS = --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w
DEEPCHECKS_SITE_PACKAGES ?= $(wildcard .venv-deepchecks/lib/python3.12/site-packages .venv-deepchecks/Lib/site-packages)
DEEPCHECKS_PYTHON ?= $(firstword $(wildcard .venv-deepchecks/bin/python .venv-deepchecks/Scripts/python.exe))

## Chequeos de seguridad: bandit (código) y pip-audit (dependencias de los dos entornos)
.PHONY: seguridad
seguridad:
	bandit -c pyproject.toml -r mibici_mlops scripts tests
	pip-audit --skip-editable $(VULNS_ACEPTADAS_PROYECTO)
	pip-audit --path $(DEEPCHECKS_SITE_PACKAGES) $(VULNS_ACEPTADAS_DEEPCHECKS)

## Validación de datos con DeepChecks (P1, P2 y P3; requiere .venv-deepchecks y data/processed)
.PHONY: validar_datos
validar_datos:
	$(DEEPCHECKS_PYTHON) scripts/validar_datos_p1.py
	$(DEEPCHECKS_PYTHON) scripts/validar_datos_p2.py
	$(DEEPCHECKS_PYTHON) scripts/validar_datos_p3.py

## Todo: pruebas, seguridad y validación de datos
.PHONY: checks
checks: test seguridad validar_datos


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
