
# coding: utf-8

# In[20]:

def richardsCurve(Velocity,B,M,g):
    return 1200*(1+.1835*np.exp(-1*B*(Velocity-M)))**(-1/g)


# In[21]:

import scipy.optimize
import numpy as np

velocities = np.array([0,.5,1,1.5,2,2.5,3,3.5,4,10])
power = np.array([0,20,75,300,800, 1100, 1175, 1195, 1200,1200])

starting_guess = (0.1031, 18.2, 0.027)

optimized_parameters, covariance = scipy.optimize.curve_fit(richardsCurve, 
                                                                 xdata = velocities, 
                                                                 ydata = power, 
                                                                 p0 = starting_guess)


# In[22]:

optimized_parameters


# In[23]:

x = np.linspace(0,4)
y = richardsCurve(x, *optimized_parameters)

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

plt.plot(x,y)


# In[24]:

y = richardsCurve(2, *optimized_parameters)


# In[25]:

y


# In[ ]:



