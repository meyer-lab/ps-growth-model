from grmodel.pymcDoseResponse import doseResponseModel

M = doseResponseModel()

M.importData()

M.build_model()

M.sample()
