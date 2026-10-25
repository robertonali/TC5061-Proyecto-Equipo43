from loguru import logger
from pathlib import Path

import pandas as pd

import folium
import mlflow
import typer

from mibici_mlops.config import ESTACIONES_PATH, FIGURES_DIR, P3_CLUSTERS_PATH, P3_PERFILES_PATH

app = typer.Typer()

class P3Plotter:
    COLORES = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'gray', 'pink', 'black']
    COLORES_ESP = ['Rojo', 'Azul', 'Verde', 'Morado', 'Naranja', 'Amarillo', 'Café', 'Gris', 'Rosa', 'Negro']

    def __init__(self, clusters, perfiles):
        self.clusters = clusters.copy()
        self.perfiles = perfiles.copy()

        self.asignaciones = self.clusters[clusters.conjunto == 'validacion']
        self.perfiles = self.perfiles[perfiles.conjunto == 'validacion']

        self.perfiles_con_cluster = self.perfiles.merge(
            self.asignaciones[['estacion', 'conjunto', 'cluster']],
            on=['estacion', 'conjunto'],
            validate='one_to_one',
        )

        self.tabla_perfiles = self._generar_tabla_perfiles()
        self.mapa = self._generar_mapa()

    def _generar_tabla_perfiles(self):
        grupos = sorted(self.perfiles_con_cluster["cluster"].dropna().unique())
        colores = {
            grupo: P3Plotter.COLORES_ESP[i % len(P3Plotter.COLORES_ESP)]
            for i, grupo in enumerate(grupos)
        }

        columnas_sal = [c for c in self.perfiles_con_cluster if c.startswith("sal_")]
        columnas_lle = [c for c in self.perfiles_con_cluster if c.startswith("lle_")]

        promedios_sal = self.perfiles_con_cluster.groupby("cluster")[columnas_sal].mean()
        promedios_lle = self.perfiles_con_cluster.groupby("cluster")[columnas_lle].mean()

        filas = []
        for grupo in promedios_sal.index:
            top_salidas = promedios_sal.loc[grupo].nlargest(3)
            top_llegadas = promedios_lle.loc[grupo].nlargest(3)

            filas.append({
                "grupo": grupo,
                "color": colores[grupo],
                "salidas_top": "; ".join(
                    f"{c.removeprefix('sal_')} ({v:.1%})"
                    for c, v in top_salidas.items()
                ),
                "llegadas_top": "; ".join(
                    f"{c.removeprefix('lle_')} ({v:.1%})"
                    for c, v in top_llegadas.items()
                ),
            })
        return pd.DataFrame(filas)

    def _generar_mapa(self):
        mapa = folium.Map(
            location=[self.asignaciones.estacion_lat.mean(), self.asignaciones.estacion_lon.mean()],
            zoom_start=14,
            tiles='OpenStreetMap',
        )

        grupos = sorted(self.asignaciones.cluster.dropna().unique())

        for i, grupo in enumerate(grupos):
            color = P3Plotter.COLORES[i % len(P3Plotter.COLORES)]
            capa = folium.FeatureGroup(name=f'Grupo {grupo}')
            puntos = self.asignaciones[self.asignaciones['cluster'] == grupo]

            for _, fila in puntos.iterrows():
                folium.CircleMarker(
                    location=[fila.estacion_lat, fila.estacion_lon],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    tooltip=f'Estación {fila.estacion} · Grupo {grupo}',
                ).add_to(capa)

            capa.add_to(mapa)

        folium.LayerControl().add_to(mapa)
        return mapa       
        

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.mapa_path = output_dir / 'p3_clusters.html'
        self.mapa.save(self.mapa_path)

        self.tabla_perfiles_path = output_dir / 'p3_perfiles_por_grupo.csv'
        self.tabla_perfiles.to_csv(self.tabla_perfiles_path, index=False)

@app.command()
def p3(
    clusters_path: Path = P3_CLUSTERS_PATH,
    profiles_path: Path = P3_PERFILES_PATH,
    output_dir: Path = FIGURES_DIR / 'p3',
    run_id: str = None
):
    clusters = pd.read_parquet(clusters_path)
    perfiles = pd.read_parquet(profiles_path)

    logger.info('Generando gráficas y mapas de los clusters de P3...')
    p3plt = P3Plotter(clusters, perfiles)
    p3plt.save(output_dir)

    logger.success('Gráficas y mapas de los clusters de P3 generadas.')

    if (run_id):
        mlflow.log_artifact(p3plt.mapa_path, artifact_path="p3", run_id=run_id)
        mlflow.log_artifact(p3plt.tabla_perfiles_path, artifact_path="p3", run_id=run_id)

if __name__ == '__main__':
    app()
