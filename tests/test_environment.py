"""Pruebas del entorno: que el Python y los paquetes instalados sean los que declara el proyecto.

Si fallan, el resto de las pruebas (y los runs de MLflow) no son confiables: se estarían corriendo
con versiones distintas a las fijadas en requirements.txt y python_env.yaml.
"""

from importlib import import_module, metadata
from importlib.util import find_spec
from pathlib import Path
import pkgutil
import re
import subprocess  # nosec B404
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest
import yaml

PROJ_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJ_ROOT / "requirements.txt"
PYTHON_ENV = PROJ_ROOT / "python_env.yaml"
MLPROJECT = PROJ_ROOT / "MLproject"


def requisitos_fijados() -> list[Requirement]:
    """Requisitos de requirements.txt, sin comentarios ni opciones de pip (como `-e .`)."""
    lineas = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    limpias = (linea.split("#", 1)[0].strip() for linea in lineas)
    return [Requirement(linea) for linea in limpias if linea and not linea.startswith("-")]


def test_version_de_python_coincide_con_python_env():
    """Mismo major.minor que python_env.yaml; el parche (3.13.x) puede variar entre equipos."""
    esperada = yaml.safe_load(PYTHON_ENV.read_text(encoding="utf-8"))["python"]
    mayor, menor = (int(parte) for parte in str(esperada).split(".")[:2])
    assert sys.version_info[:2] == (mayor, menor), (
        f"python_env.yaml pide Python {esperada}, pero las pruebas corren con "
        f"{sys.version.split()[0]} ({sys.executable})"
    )


def test_todos_los_requisitos_estan_fijados():
    """Cada paquete de requirements.txt debe tener versión exacta (==) para ser reproducible."""
    sin_fijar = [
        str(req)
        for req in requisitos_fijados()
        if not any(spec.operator == "==" for spec in req.specifier)
    ]
    assert not sin_fijar, f"Requisitos sin versión exacta (==): {sin_fijar}"


@pytest.mark.parametrize("requisito", requisitos_fijados(), ids=lambda req: req.name)
def test_version_instalada_coincide_con_requirements(requisito: Requirement):
    try:
        instalada = metadata.version(canonicalize_name(requisito.name))
    except metadata.PackageNotFoundError:
        pytest.fail(f"{requisito.name} no está instalado; corre `pip install -r requirements.txt`")
    assert requisito.specifier.contains(instalada, prereleases=True), (
        f"{requisito.name}: instalado {instalada}, requirements.txt pide {requisito.specifier}"
    )


def test_dependencias_instaladas_son_compatibles():
    """Equivalente a `pip check`: ningún paquete instalado tiene dependencias rotas."""
    # Comando fijo, sin entrada del usuario: se marca para que bandit no lo reporte.
    resultado = subprocess.run(  # nosec B603
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr


def test_paquete_instalado_apunta_a_este_repositorio():
    """`-e .` debe dejar mibici_mlops apuntando a esta carpeta, no a una copia temporal."""
    import mibici_mlops

    ubicacion = Path(mibici_mlops.__file__).resolve()
    assert ubicacion.is_relative_to(PROJ_ROOT), (
        f"mibici_mlops se importa desde {ubicacion}, fuera de {PROJ_ROOT}; "
        "reinstala con `pip install -e .`"
    )


def modulos_del_paquete() -> list[str]:
    import mibici_mlops

    return [
        info.name for info in pkgutil.walk_packages(mibici_mlops.__path__, prefix="mibici_mlops.")
    ]


@pytest.mark.parametrize("modulo", modulos_del_paquete())
def test_modulos_del_paquete_se_importan(modulo: str):
    """Detecta imports rotos (dependencia faltante o incompatible) antes de correr un pipeline."""
    import_module(modulo)


def modulos_del_mlproject() -> list[str]:
    """Módulos que lanzan los entry points (`python -m ...`), expandiendo `{question}`."""
    comandos = re.findall(r'command:\s*"python -m ([\w.{}]+)', MLPROJECT.read_text("utf-8"))
    modulos = set()
    for comando in comandos:
        if "{question}" in comando:
            modulos.update(comando.replace("{question}", p) for p in ("p1", "p2", "p3"))
        else:
            modulos.add(comando)
    return sorted(modulos)


@pytest.mark.parametrize("modulo", modulos_del_mlproject())
def test_entry_points_del_mlproject_existen(modulo: str):
    assert find_spec(modulo) is not None, f"MLproject llama a {modulo}, que no existe"
