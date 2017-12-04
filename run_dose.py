from grmodel.pymcDoseResponse import doseResponseModel
from grmodel.pymcDoseResponse import save, read
import matplotlib.pyplot as plt

filename = "pickle_file.p"

#M = doseResponseModel()

#M.importData()

#M.sample()

#M.plot()

#save(M, filename)

M = read(filename)

M.plot()

plt.show()
