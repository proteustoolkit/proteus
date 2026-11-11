#!/usr/bin/env python3
"""
Generate a minimal Henry-domain schematic in the style of the
variable-density water-table figure.

This version avoids matplotlib so it can run in minimal environments.
It writes a tiny LaTeX picture, renders it to PDF with pdflatex, and
converts the result to PNG with pdftoppm.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile


OUT_PATH = Path(__file__).resolve().parent / "domain_henry_minimal.png"


TEX_SOURCE = r"""
\documentclass{article}
\usepackage[paperwidth=340mm,paperheight=175mm,margin=0mm]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{xcolor}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\definecolor{henryblue}{RGB}{65,105,225}
\begin{document}
\setlength{\unitlength}{1mm}
\noindent
\begin{picture}(340,175)
  % Main domain
  \linethickness{0.6mm}
  \put(50,45){\line(1,0){200}}
  \put(50,145){\line(1,0){200}}
  \put(50,45){\line(0,1){100}}
  \put(250,45){\line(0,1){100}}
  \linethickness{0.4mm}

  % Top and bottom labels
  \put(150,153){\makebox(0,0){no-flow}}
  \put(150,40){\makebox(0,0){no-flow}}

  % Initial conditions in the middle of the domain
  \put(150,103){\makebox(0,0){Flow IC: $h_0=L_y-y$}}
  \put(150,93){\makebox(0,0){Transport IC: $c_0=0$}}

  % Left inflow arrows
  \color{henryblue}
  \multiput(32,62)(0,16){5}{\vector(1,0){16}}

  % Left boundary text
  \color{black}
  \put(8,126){\makebox(0,0)[l]{Freshwater inflow}}
  \put(8,116){\makebox(0,0)[l]{$q_n=-6.6\times 10^{-5}\ \mathrm{m/s}$}}
  \put(22,103){\makebox(0,0)[l]{$c=0$}}

  % Right boundary arrow and labels
  \put(286,110){\vector(-1,0){28}}
  \put(290,128){\makebox(0,0)[l]{Saltwater boundary}}
  \put(290,117){\makebox(0,0)[l]{$h=\dfrac{\rho_s}{\rho_f}(L_y-y)$}}
  \put(290,104){\makebox(0,0)[l]{$c=1$}}

  % Bottom dimension line
  \put(50,20){\vector(1,0){100}}
  \put(250,20){\vector(-1,0){100}}
  \put(50,14){\line(0,1){12}}
  \put(250,14){\line(0,1){12}}
  \put(150,28){\makebox(0,0){$2.0~\mathrm{m}$}}

  % Left dimension line
  \put(20,45){\vector(0,1){50}}
  \put(20,145){\vector(0,-1){50}}
  \put(14,45){\line(1,0){12}}
  \put(14,145){\line(1,0){12}}
  \put(14,95){\makebox(0,0){\rotatebox{90}{$1.0~\mathrm{m}$}}}

\end{picture}
\end{document}
"""


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required tool '{name}' was not found")


def run(cmd, cwd):
    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main():
    require_tool("pdflatex")
    require_tool("pdftoppm")

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        tex_path = tmpdir / "domain_henry_minimal.tex"
        pdf_path = tmpdir / "domain_henry_minimal.pdf"
        ppm_prefix = tmpdir / "domain_henry_minimal"

        tex_path.write_text(TEX_SOURCE, encoding="utf-8")

        run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tmpdir,
        )

        run(
            ["pdftoppm", "-png", "-singlefile", "-r", "170", pdf_path.name, ppm_prefix.name],
            cwd=tmpdir,
        )

        png_path = tmpdir / "domain_henry_minimal.png"
        OUT_PATH.write_bytes(png_path.read_bytes())

    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
