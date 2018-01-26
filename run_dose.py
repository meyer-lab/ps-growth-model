
from grmodel.pymcDoseResponse import doseResponseModel
from grmodel.pymcDoseResponse import save, readSamples
import matplotlib.pyplot as plt
from pymc3.backends.tracetab import trace_to_dataframe
from sklearn import preprocessing
import seaborn as sns


M = doseResponseModel()

M.importData()

M.sample()

save(M, "sampling.pkl")

'''

M = readSamples()
M.traceplot()

df = readSamples(asdf=True)

df.drop(list(df.filter(regex='lnum')), axis=1, inplace=True)
df.drop(list(df.filter(regex='lExp')), axis=1, inplace=True)
df.drop(list(df.filter(regex='apop')), axis=1, inplace=True)

sns.pairplot(df)

plt.show()
'''
