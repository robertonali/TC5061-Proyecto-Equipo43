"""Chequeos de seguridad básicos del repositorio.

Complementan a bandit (análisis estático del código) y pip-audit (vulnerabilidades conocidas en
las dependencias), que se corren aparte (ver "Chequeos de seguridad" en el README):

- Los archivos con secretos (.env, el secreto de DVC, llaves y credenciales) están en .gitignore.
- Ningún archivo versionado contiene algo con forma de secreto.
- `.dvc/config`, que sí se sube, no guarda el secreto del cliente de Google Drive.
- La descarga de datos solo abre URL http(s) (el aviso B310 de bandit).
"""

from pathlib import Path
import re
import shutil
import subprocess  # nosec B404

import pytest

from mibici_mlops import dataset

PROJ_ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")

# Patrones de secretos comunes. Cada uno se prueba abajo con un ejemplo, para que una regex rota
# no haga pasar la prueba sin revisar nada.
PATRONES_SECRETOS = {
    "secreto de cliente OAuth de Google": re.compile(r"GOCSPX-[A-Za-z0-9_\-]{20,}"),
    "llave de API de Google": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "llave de acceso de AWS": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "token de GitHub": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "llave privada": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    "contraseña escrita en el código": re.compile(
        r"""(?i)\b(?:password|passwd|contrase(?:ñ|n)a|secret|token)\s*[:=]\s*["'][^"'\s]{8,}["']"""
    ),
}

requiere_git = pytest.mark.skipif(
    GIT is None or not (PROJ_ROOT / ".git").exists(), reason="requiere git y el repositorio"
)


def git(*argumentos: str) -> subprocess.CompletedProcess:
    # Git con argumentos fijos, sin shell.
    return subprocess.run(  # nosec B603
        [GIT, *argumentos], cwd=PROJ_ROOT, capture_output=True, text=True, check=False
    )


def archivos_versionados_de_texto() -> list[Path]:
    """Archivos que sigue Git, sin binarios ni reportes generados (HTML de DeepChecks)."""
    rutas = git("ls-files").stdout.splitlines()
    texto = []
    for ruta in rutas:
        archivo = PROJ_ROOT / ruta
        if not archivo.is_file() or archivo.suffix in {".html", ".png", ".pdf", ".jpg"}:
            continue
        contenido = archivo.read_bytes()
        if b"\0" not in contenido[:8192]:
            texto.append(archivo)
    return texto


@pytest.mark.parametrize(
    ("nombre", "ejemplo"),
    [
        ("secreto de cliente OAuth de Google", "GOCSPX-" + "a1B2c3D4e5F6g7H8i9J0kLmN"),
        ("llave de API de Google", "AIza" + "S" * 35),
        ("llave de acceso de AWS", "AKIA" + "ABCDEFGHIJKLMNOP"),
        ("token de GitHub", "ghp_" + "a" * 36),
        ("llave privada", "-----BEGIN " + "RSA PRIVATE KEY-----"),
        ("contraseña escrita en el código", 'password = "' + "hunter22hunter" + '"'),
    ],
)
def test_patrones_de_secretos_detectan_su_ejemplo(nombre: str, ejemplo: str):
    """Caso normal: cada patrón reconoce un secreto falso con su forma (armado por partes para
    que este archivo no se detecte a sí mismo)."""
    assert PATRONES_SECRETOS[nombre].search(ejemplo)


@pytest.mark.parametrize(
    "texto",
    [
        'MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")',
        "dvc remote modify --local drive gdrive_client_secret <secreto>",
        "gdrive_client_id = 1065382686975-xxxx.apps.googleusercontent.com",
        'token = ""',
    ],
)
def test_patrones_de_secretos_ignoran_texto_normal(texto: str):
    """Caso límite: placeholders, variables de entorno y el client_id público no son secretos."""
    assert not any(patron.search(texto) for patron in PATRONES_SECRETOS.values())


@requiere_git
def test_ningun_archivo_versionado_tiene_secretos():
    encontrados = [
        f"{archivo.relative_to(PROJ_ROOT)}: {nombre}"
        for archivo in archivos_versionados_de_texto()
        for nombre, patron in PATRONES_SECRETOS.items()
        if patron.search(archivo.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not encontrados, f"Posibles secretos en archivos versionados: {encontrados}"


@requiere_git
@pytest.mark.parametrize(
    "ruta",
    [
        ".env",
        ".env.local",
        ".dvc/config.local",
        ".dvc/tmp/gdrive-user-credentials.json",
        "credentials.json",
        "client_secret_123.json",
        "llave.pem",
        "certificado.p12",
        "mlflow.db",
        "data/processed/p1_estacion_dia.parquet",
    ],
)
def test_archivos_sensibles_estan_en_gitignore(ruta: str):
    """Secretos, credenciales, la base de MLflow y los datos procesados no se pueden subir."""
    assert git("check-ignore", "-q", ruta).returncode == 0, f"{ruta} no está en .gitignore"


@requiere_git
def test_plantilla_env_example_si_se_versiona():
    """Caso límite del patrón `.env.*`: la plantilla sin secretos sí debe poder subirse."""
    assert git("check-ignore", "-q", ".env.example").returncode == 1
    assert (PROJ_ROOT / ".env.example").is_file()


def test_dvc_config_no_guarda_el_secreto_de_google_drive():
    """El secreto va en .dvc/config.local (`dvc remote modify --local`), nunca en .dvc/config."""
    config = (PROJ_ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    assert "gdrive_client_secret" not in config


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://www.mibici.net/datos.csv",
        "/site/assets/files/datos.csv",
        "javascript:alert(1)",
    ],
)
def test_abrir_url_rechaza_esquemas_distintos_de_http(url: str):
    with pytest.raises(ValueError, match="http"):
        dataset.abrir_url(url, timeout=1)


@pytest.mark.parametrize("url", ["https://www.mibici.net/x.csv", "http://127.0.0.1:5000/health"])
def test_abrir_url_acepta_http_y_https(monkeypatch, url: str):
    """Caso normal, sin red: se sustituye `urlopen` y se revisa la petición que recibe."""
    peticiones = []
    monkeypatch.setattr(
        dataset, "urlopen", lambda peticion, timeout: peticiones.append((peticion, timeout))
    )
    dataset.abrir_url(url, timeout=7)
    assert peticiones[0][0].full_url == url
    assert peticiones[0][1] == 7
