# The paper's figures

One script per scripted figure. Figs. 1, 3 and 7 are drawn by hand and have no
script. Output is written to `figures/` as PDF and PNG.

| paper | script |
|---|---|
| Fig. 2 | `fig2_link_primitives.py` |
| Fig. 4 | `fig4_actuator_trends.py` |
| Fig. 5 | `fig5_twin_dynamics.py` |
| Fig. 6 | `fig6_quadruped_lineup.py` |
| Fig. 8 | `fig8_capability.py` |

No figure fits anything; each reads a JSON another stage wrote, or a built model.

## Redrawing

```shell
pip install -e ".[figures,twins]" -c constraints.txt
python scripts/figures/make_all.py
```

What each needs first:

```shell
# Fig. 2 — the models the primitives are cut from
python scripts/generate_robot.py --robot humanoid  --output-dir generated/figlinks      --fixed-output-dir
python scripts/generate_robot.py --robot quadruped --output-dir generated/figlinks_quad --fixed-output-dir

# Fig. 4 — the leave-one-out record
python datasets/actuators/scripts/05_leave_one_out.py

# Fig. 5 — the twins and their paired vendor models
python scripts/setup_data.py && python scripts/make_twins.py

# Fig. 6 — the three designs, in one scene
python scripts/generate_quadrupeds.py
python scripts/figures/render_lineup_plate.py --out generated/plots/paper_lineup.png

# Fig. 8 — nothing; it reads experiments/results.json
```

Figs. 2, 5 and 6 render through viser in headless Chrome, so Chrome or Chromium
must be installed.

Labels are typeset by LaTeX, as in the paper, when a full TeX is installed (on
Debian/Ubuntu: `texlive-latex-extra`, `cm-super`, `dvipng`). Without one the
figures fall back to matplotlib's mathtext and say so. `make_all.py` draws what
it can and lists any figure whose inputs are missing.

Style lives in `paper_style.py`: scienceplots `science`+`ieee`, IEEE column
widths, Okabe-Ito colours, vector PDF plus 600-dpi PNG. The catalogue's
`Harmonic` category is labelled **High GR** in every figure.
