import numpy as np
import pandas as pd

filename = '/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/test/event000000000-cells.csv'

df = pd.DataFrame(np.arange(4).reshape((1,4)), columns=['hi_id', 'ch0','ch1','value'])
#print(df)
#    A  B
# 0  0  1
# 1  2  3
# 2  4  5
# 3  6  7
# 4  8  9

# Save to HDF5
df.to_hdf(filename, 'data', mode='w', format='table')
del df    # allow df to be garbage collected

# Append more data
#df2 = pd.DataFrame(np.arange(10).reshape((5,2))*10, columns=['A', 'B'])
#df2.to_hdf(filename, 'data', append=True)

#print(pd.read_hdf(filename, 'data'))