#!/usr/bin/env python3

import sys
import matplotlib as plt

plt.use("AGG")
from grmodel.figures.FigureCommon import overlayCartoon

fdir = "./Manuscript/Figures/"

if __name__ == "__main__":
    nameOut = "Figure" + sys.argv[1]

    exec("from grmodel.figures import " + nameOut)
    ff = eval(nameOut + ".makeFigure()")

    print(fdir + nameOut + ".svg")

    ff.savefig(fdir + nameOut + ".svg", dpi=ff.dpi, bbox_inches="tight", pad_inches=0)

    if sys.argv[1] == "1":
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + "Figure1.svg", "./grmodel/figures/Figure1-Schematic.svg", 50, 25, 0.62)
    elif sys.argv[1] == "2":
        # Overlay Figure 2 cartoon
        overlayCartoon(fdir + "Figure2.svg", "./grmodel/figures/Figure2-Schematic.svg", 560, 25, 0.6)
        overlayCartoon(fdir + "Figure2.svg", "./grmodel/figures/Figure2-CFSE.svg", 23, 450, 1.0)
        overlayCartoon(fdir + "Figure2.svg", "./grmodel/figures/Figure2-CFSE-endpoint.svg", 360, 445, 0.6)

    print(nameOut + " is done.")
