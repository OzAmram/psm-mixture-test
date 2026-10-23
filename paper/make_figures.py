"""Vector (PDF) versions of the post's figures for the paper, from the same plotting code and the same result files.

    python paper/make_figures.py          # run from the repo root -> paper/figures/*.pdf

The plotting scripts are run unchanged, with a few presentation patches for a two-panels-per-row layout at the
5.5-inch text width of the ICLR style: every savefig goes to paper/figures/<name>.pdf instead of the PNG under
results/figures/ (so the post's figures are untouched); figures are sized at 0.75 in per bar group; in-figure
titles are dropped (the LaTeX subcaptions carry them); bar labels are shortened (SHORT below) and fonts set to 7 pt.
The method diagrams are converted from their SVGs. No data or computation is changed.
"""
import runpy
from pathlib import Path
import cairosvg
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

OUT = Path("paper/figures"); FS = 7
SHORT = {"Olmo Instruct\n(generator)": "Generator", "Qwen Instruct\n(generator)": "Generator",
         "Olmo base\nPHLE": "Olmo base\nPHLE", "Qwen base\nPHLE": "Qwen base\nPHLE",
         "Prompted Olmo\nInstruct classifier": "Prompted\nOlmo Instruct", "Prompted\nGPT-4.1 classifier": "Prompted\nGPT-4.1",
         "Sibling students\n(same teachers,\ndifferent run)": "Sibling\nstudents", "Olmo Instruct": "Olmo\nInstruct", "Qwen Instruct": "Qwen\nInstruct"}
_save, _subplots, _xtl, _tp, _leg, _yl = Figure.savefig, plt.subplots, Axes.set_xticklabels, Axes.tick_params, Axes.legend, Axes.set_ylabel

def savefig(self, fn, *a, **kw):
    kw.pop("dpi", None); out = OUT / (Path(str(fn)).stem.removesuffix("_v2") + ".pdf")
    self.tight_layout(pad=0.3); _save(self, out, *a, **kw); print("->", out)

def subplots(*a, figsize=None, **kw):
    fig, ax = _subplots(*a, figsize=(3.2, 2.3), **kw); fig._paper = True; return fig, ax

def set_xticklabels(self, labels, *a, **kw):
    labels = [SHORT.get(l, l) for l in labels]; kw["fontsize"] = FS - 0.5
    w = 0.62 * len(labels) + 0.7; self.figure.set_size_inches(max(3.2, w), 2.3)
    return _xtl(self, labels, *a, **kw)

def tick_params(self, *a, **kw):
    if "labelsize" in kw: kw["labelsize"] = FS - 0.5
    return _tp(self, *a, **kw)

def legend(self, *a, **kw):
    kw["fontsize"] = FS; kw["handlelength"] = 1.2; kw["columnspacing"] = 1.0; return _leg(self, *a, **kw)

def set_ylabel(self, label, *a, **kw):
    kw["fontsize"] = FS; return _yl(self, label, *a, **kw)

Figure.savefig, plt.subplots = savefig, subplots
Axes.set_xticklabels, Axes.tick_params, Axes.legend, Axes.set_ylabel = set_xticklabels, tick_params, legend, set_ylabel
Axes.set_title = lambda self, *a, **kw: None
runpy.run_path("scripts/plot_auroc.py", run_name="__main__")
for name in ["method_diagram_v2", "method_diagram_af_v2"]:
    out = OUT / f"{name.removesuffix('_v2')}.pdf"; cairosvg.svg2pdf(url=f"results/figures/{name}.svg", write_to=str(out)); print("->", out)
