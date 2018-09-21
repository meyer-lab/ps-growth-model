#! /usr/bin/env python3
import pymc3 as pm
import matplotlib.pyplot as plt
from grmodel.interactionData import readCombo, filterDrugC, dataSplit
from grmodel.pymcInteraction import build_model

df = readCombo()

df = filterDrugC(df, 'PIM447', 'BYL749')

X1, X2, timeV, phase, red, green = dataSplit(df)

M = build_model(X1, X2, timeV, 1.0, confl=phase, apop=green, dna=red)

fit = pm.sampling.sample(model=M)

pm.plots.traceplot(fit)
plt.show()