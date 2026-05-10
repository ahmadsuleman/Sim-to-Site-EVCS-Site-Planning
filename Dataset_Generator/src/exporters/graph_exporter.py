from __future__ import annotations

from pathlib import Path
import networkx as nx
import pandas as pd


class GraphExporter:
    def __init__(self, graph_dir: Path):
        self.graph_dir = graph_dir
        self.graph_dir.mkdir(parents=True, exist_ok=True)

    def export(self, G: nx.Graph, nodes: pd.DataFrame, edges: pd.DataFrame, export_graphml: bool = True, export_gpkg_if_available: bool = True) -> None:
        if export_graphml:
            safe = G.copy()
            for _, _, attrs in safe.edges(data=True):
                for k, v in list(attrs.items()):
                    if v is None:
                        attrs[k] = ""
            nx.write_graphml(safe, self.graph_dir / "graph.graphml")
        # Always provide a portable graph table fallback.
        nodes.to_csv(self.graph_dir / "graph_nodes.csv", index=False)
        edges.to_csv(self.graph_dir / "graph_edges.csv", index=False)
        if export_gpkg_if_available:
            try:
                import geopandas as gpd  # type: ignore
                from shapely.geometry import Point, LineString  # type: ignore
                node_geom = [Point(float(r.lon), float(r.lat)) for r in nodes.itertuples()]
                gnodes = gpd.GeoDataFrame(nodes.copy(), geometry=node_geom, crs="EPSG:4326")
                node_pos = nodes.set_index("node_id")[["lat", "lon"]].to_dict("index")
                lines = []
                for r in edges.itertuples():
                    a = node_pos.get(str(r.source))
                    b = node_pos.get(str(r.target))
                    if a and b:
                        lines.append(LineString([(a["lon"], a["lat"]), (b["lon"], b["lat"])]))
                    else:
                        lines.append(None)
                gedges = gpd.GeoDataFrame(edges.copy(), geometry=lines, crs="EPSG:4326")
                gpkg = self.graph_dir / "graph.gpkg"
                gnodes.to_file(gpkg, layer="nodes", driver="GPKG")
                gedges.to_file(gpkg, layer="edges", driver="GPKG")
            except Exception as exc:
                (self.graph_dir / "graph_gpkg_unavailable.txt").write_text(
                    f"GeoPackage export requires geopandas/shapely/fiona support. Error: {exc}\n",
                    encoding="utf-8",
                )
