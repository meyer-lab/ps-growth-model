#! /usr/bin/env python3
import pymc3 as pm
import numpy as np
from grmodel.pymcInteraction import blissInteract, drugInteractionModel, build_model
from grmodel.sampleAnalysis import read_dataset
from grmodel.pymcGrowth import theanoCore
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def build(loadFile='BYLvPIM', drug1='PIM447', drug2='BYL749'):
  """ build and save the drugInteractionModel """
  DI = drugInteractionModel(loadFile, drug1, drug2)
  model = build_model(DI.X1, DI.X2, DI.timeV, 1.0, DI.phase, DI.red, DI.green)
  DI.save()


def makeFigure(loadFile='BYLvPIM'):
  """ plot the number of live cells and dead cells by time
      for different drug concentration for drug 1 and drug 2 """

  # read model from saved pickle file
  M = read_dataset(loadFile)

  # trace is drawn from 1000 pymc samplings
  trace = pm.backends.tracetab.trace_to_dataframe(M.fit)

  # traceplot
  pm.plots.traceplot(M.fit)

  # df contains the actual experimental data
  df = M.df

  def transform(name):
    """ transforms the data structure """
    return np.vstack((np.array(trace[name + '__0'])[0], np.array(trace[name + '__1'])[0]))

  E_con = transform('E_con')
  hill_death = transform('hill_death')
  hill_growth = transform('hill_growth')
  IC50_death = transform('IC50_death')
  IC50_growth = transform('IC50_growth')

  death_rates = E_con[0] * blissInteract(M.X1, M.X2, hill_death, IC50_death, numpyy=True)
  growth_rates = E_con[1] * (1 - blissInteract(M.X1, M.X2, hill_growth, IC50_growth, numpyy=True))

  # compute the number of live cells, dead cells, early apoptosis cells
  lnum, eap, deadapop, deadnec = theanoCore(M.timeV, growth_rates, death_rates,
                                            np.array(trace['apopfrac'])[0],
                                            np.array(trace['d'])[0], numpyy=True)
  dead = deadapop + deadnec

  # compute the median of corr
  median_confl_corr = np.median(np.array(trace['confl_corr']))
  median_apop_corr = np.median(np.array(trace['apop_corr']))
  median_dna_corr = np.median(np.array(trace['dna_corr']))

  # plot the graphs of the number of live cells and dead cells by time
  # for different drug concentrations
  plt.figure()
  plt.subplot(211)
  for i in range(len(M.X1)):
    plt.plot(M.timeV, lnum[i])
  plt.ylabel('The number of live cells')
  plt.title('The number of live cells and dead cells by time (' + loadFile + ')')

  # add text box for displaying the corr
  text_box = AnchoredText("median_confl_corr = %.3f" % median_confl_corr
                          + "\nmedian_apop_corr = %.3f" % median_apop_corr
                          + "\nmedian_dna_corr = %.3f" % median_dna_corr, frameon=True, loc=2, pad=0.5)
  plt.setp(text_box.patch, facecolor='white', alpha=0.5)
  plt.gca().add_artist(text_box)

  plt.subplot(212)
  for i in range(len(M.X1)):
    plt.plot(M.timeV, dead[i])
  plt.ylabel('The number of dead cells')
  plt.xlabel('Time (hours)')

  plt.show()


# build(loadFile='OSIvBIN', drug1='Binimetinib', drug2='OSI-906')
files = ['BYLvPIM', 'OSIvBIN', 'LCLvTXL']
for ff in files:
  makeFigure(loadFile=ff)
