import pandas as pd
data_confl = pd.read_csv("./091916_H1299_cytotoxic_confluence_confl.csv", infer_datetime_format=True)
#data_confl.shape #(111, 21)
data_confl = data_confl.ix[:, 0:19] #remove "blank" column
data_green = pd.read_csv("./091916_H1299_cytotoxic_confluence_green.csv", infer_datetime_format=True)
#data_green.shape #(111, 21)
data_green = data_green.ix[:, 0:19]
