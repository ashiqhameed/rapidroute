import io
import urllib.parse
import urllib.request
import zipfile

from src.config import CFG, path

EMS_ENDPOINT = "https://data.cityofnewyork.us/resource/76xm-jjuj.csv"
ZCTA_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip"


def fetch_ems():
    out = path("ems_raw")
    if out.exists():
        print(f"[skip] {out.name} exists")
        return
    y = CFG["year"]
    query = {
        "$where": f"borough='{CFG['borough']}' AND incident_datetime between '{y}-01-01T00:00:00' and '{y}-12-31T23:59:59'",
        "$order": "incident_datetime",
        "$limit": "1000000",
    }
    url = EMS_ENDPOINT + "?" + urllib.parse.urlencode(query)
    print("[get] EMS incidents")
    urllib.request.urlretrieve(url, out)


def fetch_zcta():
    out = path("zcta_raw")
    if out.exists():
        print(f"[skip] {out.name} exists")
        return
    print("[get] ZCTA gazetteer")
    with urllib.request.urlopen(ZCTA_URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(out.parent)


def finish_graph(G, out):
    import osmnx as ox

    G = ox.truncate.largest_component(G, strongly=True)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, out)
    print(f"[ok] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def fetch_graph():
    out = path("graph_raw")
    if out.exists():
        print(f"[skip] {out.name} exists")
        return
    import osmnx as ox

    xml = out.parent / "manhattan.osm"
    if xml.exists():
        print(f"[build] road graph from {xml.name}")
        finish_graph(ox.graph_from_xml(xml, bidirectional=False, simplify=True, retain_all=False), out)
        return
    try:
        print(f"[get] road graph: {CFG['place']}")
        finish_graph(ox.graph_from_place(CFG["place"], network_type="drive"), out)
    except Exception as e:
        print(f"[fail] OSMnx download failed: {type(e).__name__}")
        print("Run this, then re-run the script:")
        print(f'curl -L -o "{xml}" --data-urlencode data@src/data/manhattan_drive.overpassql https://overpass-api.de/api/interpreter')


if __name__ == "__main__":
    fetch_ems()
    fetch_zcta()
    fetch_graph()
