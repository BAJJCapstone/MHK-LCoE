
# coding: utf-8

# In[1]:

from NOAAStations import TidalStation
from DeviceModels import Turbine, calculate_power

import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt


# In[2]:

MCT = Turbine(1200., 0.1835, 3.55361367,  2.30706792,  1.05659521)

Sagamore = TidalStation(8447173)

results, times = calculate_power(Sagamore, MCT, 0, 0, 365*24*3600, 9.8, 3)


# In[4]:

plt.plot(times/(24*3600), results/1000)
plt.xlim(0,365)
plt.ylabel('Energy (MJ)')
plt.xlabel('Time (day)')


# In[ ]:



