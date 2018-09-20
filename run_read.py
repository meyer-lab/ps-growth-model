#! /usr/bin/env python3
import numpy as np
import pymc3 as pm
from grmodel.interactionData import readCombo, filterDrugC, dataSplit
from grmodel.pymcInteraction import build_model

df = readCombo()

df = filterDrugC(df, 'PIM447', 'BYL749')

X1, X2, timeV, phase, red, green = dataSplit(df)

M = build_model(X1, X2, timeV, 1.0, confl=phase, apop=green, dna=red)

outt = pm.find_MAP(model=M, method='L-BFGS-B')

print(outt)