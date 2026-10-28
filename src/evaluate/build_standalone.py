import base64
import re

from src.config import ROOT

WEB = ROOT / "web"
FIGS = ROOT / "reports" / "figures"


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    html = (WEB / "index.html").read_text()
    css = (WEB / "style.css").read_text()
    data = (WEB / "data.js").read_text()
    app = (WEB / "app.js").read_text()

    used = set(re.findall(r"reports/figures/([\w\-]+\.png)", html)) | set(re.findall(r'img: "([\w\-]+\.png)"', app))
    figs = {name: data_uri(FIGS / name) for name in sorted(used)}
    for name, uri in figs.items():
        html = html.replace(f"../reports/figures/{name}", uri)

    figs_js = "window.RR_FIGS = {" + ",".join(f'"{k}":"{v}"' for k, v in figs.items()) + "};"
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="data.js"></script>', f"<script>{data}</script>\n  <script>{figs_js}</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>\n{app}\n</script>")

    out = ROOT / "RapidRoute.html"
    out.write_text(html)
    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB, {len(figs)} figures embedded)")


if __name__ == "__main__":
    main()
