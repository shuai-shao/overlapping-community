# main file for executing, inputing and outputing results.
import sys
import numpy as np
import pandas as pd 
import sync_class as sc
reload(sc)
import matplotlib.pyplot as plt
from build_network import *

ts = 0.6
adjacency = stocks_nw(ts)

array_a = np.arange(35) + 91
array_b = np.arange(71) + 126
array_total = np.arange(106) + 91
ss = sc.Osc(adjacency,array_a,array_b)    # create an object from class Osc

end_time = 3000   # time length for running simulation
phi_dot_matrix= np.zeros([end_time,ss.n])    # matrix for organizing and outputing data

ss.coef()
for i in range(end_time):
    ss.update()
    phi_dot_matrix[i,:] = np.copy(ss.phi_dot)    #write phi_dot to phi_dot_matrix

array_nonzero = np.intersect1d(array_total, np.arange(ss.n)[ss.k!=0])

plt.plot(phi_dot_matrix[-1000:,array_nonzero])
plt.show()


#phi_dot_frame = pd.DataFrame(phi_dot_matrix)    #convert to pandas dataframe, easier to write to csv files
#phi_dot_frame.to_csv('phi_dot.csv',index = False)
