"""Vector (PDF) versions of the post's figures for the paper, from the same plotting code and the same result files.

    python paper/make_figures.py          # run from the repo root -> paper/figures/*.pdf

The plotting scripts are run unchanged, with two patches: every savefig goes to paper/figures/<name>.pdf instead of the
PNG under results/figures/ (so the post's figures are untouched), and every figure is made at SCALE times its size so the
fixed font sizes stay legible when two panels sit side by side at half the text width. The method diagrams are converted
from their SVGs.
"""
import runpy
from pathlib import Path
import cairosvg
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

OUT = Path("paper/figures"); SCALE = 0.78
_save, _subplots = Figure.savefig, plt.subplots

def savefig(self, fn, *a, **kw):
    kw.pop("dpi", None); out = OUT / (Path(str(fn)).stem.removesuffix("_v2") + ".pdf")
    _save(self, out, *a, **kw); print("->", out)

def subplots(*a, figsize=None, **kw):
    w, h = figsize or plt.rcParams["figure.figsize"]
    return _subplots(*a, figsize=(w * SCALE, h * SCALE), **kw)

Figure.savefig, plt.subplots = savefig, subplots
runpy.run_path("scripts/plot_auroc.py", run_name="__main__")
for name in ["method_diagram_v2", "method_diagram_af_v2"]:
    cairosvg.svg2pdf(url=f"results/figures/{name}.svg", write_to=str(OUT / f"{name.removesuffix('_v2')}.pdf")); print("->", OUT / f"{name}.pdf")
