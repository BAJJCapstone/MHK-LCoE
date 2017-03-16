
# coding: utf-8

# In[1]:

from NOAAStations import TidalStation
from DeviceModels import Turbine, calculate_power
from DeviceModels import power

import numpy as np
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt


# In[2]:

MCT = Turbine(1200., 0.1835, 0.1031, 18.2, 0.027)
Sagamore = TidalStation(8447173)

results = calculate_power(Sagamore, MCT, 0, 0, 365*24, 9.8, 3)


# In[3]:

plt.plot(results)


# In[4]:

Sagamore.graphHarmonicConstituent(0, 10000)


# In[ ]:



