"""Punto de entrada `train` del MLproject: entrena el modelo de la pregunta elegida.

`mlflow run . -e train -P question=p2 -P model=hgb -P learning_rate=0.05` llama a este módulo con
`--question p2 --model hgb --learning_rate 0.05`: MLflow agrega como `--nombre valor` los
parámetros que el MLproject no declara (los hiperparámetros). Aquí se pasan al script de la
pregunta, con el nombre como lo espera typer (`--learning-rate`), y `--model` como `--modelo`.

Para conectar una pregunta, agrega su módulo a PREGUNTAS: debe tener un `app` de typer con la
opción `--modelo`. Los valores por defecto los pone el script de la pregunta desde config.py.

Uso directo (sin MLflow Project):
    python -m mibici_mlops.modeling.train --question p2 --model frecuencia
"""

import importlib

import typer

app = typer.Typer()

# Pregunta → módulo con su entrenamiento. P1 y P3 se agregan cuando sus scripts acepten --modelo.
PREGUNTAS = {
    "p2": "mibici_mlops.modeling.train_p2",
    "p3": "mibici_mlops.modeling.train_p3"
}


def nombre_de_opcion(argumento: str) -> str:
    """`--max_iter` → `--max-iter` y `--C` → `--c`: de los nombres de -P a opciones de typer."""
    if argumento.startswith("--"):
        return argumento.lower().replace("_", "-")
    return argumento


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    question: str = typer.Option(..., help=f"Pregunta: {', '.join(PREGUNTAS)}."),
    model: str = typer.Option(..., help="Modelo de esa pregunta (ver su script)."),
):
    if question not in PREGUNTAS:
        raise typer.BadParameter(
            f"conectadas al MLproject: {', '.join(PREGUNTAS)}", param_hint="--question"
        )
    modulo = importlib.import_module(PREGUNTAS[question])
    argumentos = ["--modelo", model, *map(nombre_de_opcion, ctx.args)]
    modulo.app(argumentos, prog_name=f"train --question {question}")


if __name__ == "__main__":
    app()
