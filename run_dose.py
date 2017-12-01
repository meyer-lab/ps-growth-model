from grmodel.pymcDoseResponse import doseResponseModel
from grmodel.pymcDoseResponse import save, read

filename = "pickle_file.p"

M = doseResponseModel()

M.importData()

M.sample()

M.plot()

save(M, filename)

#read(filename)
