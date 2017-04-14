
# coding: utf-8

# In[1]:

import scipy.optimize
import numpy as np
import pandas as pd

import os
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
figsize(12, 10)
sbn.set_context("paper", font_scale=1)
sbn.set_style("whitegrid")


from NOAAStations import TidalStation
from DeviceModels import Turbine, calculate_power
from Calculator import maintenance, operation


# In[13]:

#COI0508
# station_data = os.path.join('currentData', '{}.pkl'.format(station_id))
max_average = 0 
failures = 0
for pkl_file in os.listdir(os.path.join('currentData')):
    station_id = pkl_file.split('.')[0]
    currents = pd.read_pickle(os.path.join('currentData', pkl_file))
    currents.dropna()
    for column in currents:
        try:
            if column.endswith('.s'): 
                average = pd.to_numeric(currents[column]).mean()
            else: 
                continue
        except TypeError:
            failures += 1
            print('uh oh')
            continue
        if float(average) > float(max_average):
            max_average = average
            location = column   
print('Found max average of {} in {}'.format(float(max_average)/100, location))
# speedAndDirection = pd.DataFrame(currents[speed_string].values/100.*np.cos(currents[direction_string].values*np.pi/180.), 
#                                  index=currents.index)
                                 
# plt.figure()
# speedAndDirection.plot()
# plt.show()


# In[14]:

station_id = location.split('.')[0]
station_data = os.path.join('currentData', '{}.pkl'.format(station_id))

currents = pd.read_pickle(station_data)
currents.dropna()
bin_number = location.split('.')[1]

currents['{}.{}.s'.format(station_id, bin_number)] = pd.to_numeric(currents['{}.{}.s'.format(station_id, bin_number)])
currents['{}.{}.d'.format(station_id, bin_number)] = pd.to_numeric(currents['{}.{}.d'.format(station_id, bin_number)])
speedAndDirection = pd.DataFrame(currents['{}.{}.s'.format(station_id, bin_number)].values/100.*np.cos(currents['{}.{}.d'.format(station_id, bin_number)].values*np.pi/180.), 
                                 index=currents.index)
                                 
plt.figure()
plt.plot(np.arange(0, len(currents['{}.{}.d'.format(station_id, bin_number)])),np.cos(currents['{}.{}.d'.format(station_id, bin_number)]*np.pi/180), 'o')
plt.show()


# In[15]:

Anchor_Point = TidalStation(9455606)
time, height = Anchor_Point.predictWaterLevels(0, 24*30)
height_constituents = Anchor_Point.constituents



# In[16]:

def set_up_least_squares(constituents, **height_constituents):
    hm = {key: float(dicts['Speed']) for key, dicts in height_constituents.items()}
    def harmonicConstituentModel(time, *amp_and_phase):
        assert len(amp_and_phase) // 2 == len(height_constituents.keys())
        assert len(amp_and_phase) % 2 == 0
        velocity = 0 
        for i, constituent in enumerate(constituents):
            velocity += amp_and_phase[2*i]*np.cos((hm[constituent] * time + amp_and_phase[2*i+1])*np.pi/180.)
        return velocity    
    return harmonicConstituentModel

velocities = speedAndDirection.as_matrix()
time = np.arange(0, len(velocities))*6/60
data = np.column_stack((time, velocities[:,0]))
data = data[~np.isnan(data).any(axis=1)]
upper_bounds = []
starting_guess = []
constituents = []
for keys, dicts in height_constituents.items():
    starting_guess.append(float(dicts['Amplitude']))
    upper_bounds.append(np.inf)
    if float(dicts['Phase'])+180 < 360: starting_guess.append(float(dicts['Phase']) + 180)
    else: starting_guess.append(float(dicts['Phase']) - 180)
    upper_bounds.append(360)
    constituents.append(keys)
    
lower_bounds = [0]*len(upper_bounds)    
param_bounds = (lower_bounds, upper_bounds)
starting_guess = tuple(starting_guess)




optimized_parameters, covariance = scipy.optimize.curve_fit(set_up_least_squares(constituents, **height_constituents), 
                                                             xdata = data[:,0], 
                                                             ydata = data[:,1],
                                                             bounds = param_bounds,
                                                             p0 = starting_guess)

        
print(optimized_parameters)


# In[17]:

print(constituents)

with open('HM-{}-{}.txt'.format(station_id, 38),'w') as myFile:
    for i, constituent in enumerate(constituents):
        myFile.write('{},{},{}\n'.format(optimized_parameters[2*i],height_constituents[constituent]['Speed'], optimized_parameters[2*i+1]))
    


# In[20]:

def harmonicConstituentModel(time, *hm):
    assert len(hm) % 3 == 0
    velocity = 0 
    for i in range(len(hm)//3):
        velocity += hm[3*i]*np.cos((hm[3*i+1] * time + hm[3*i+2])*np.pi/180.)
    return velocity


plt.figure()
speedAndDirection.plot()
plt.show()

velocities = speedAndDirection.as_matrix()

time = np.arange(0, len(velocities))*6/60
data = np.column_stack((time, velocities[:,0]))
data = data[~np.isnan(data).any(axis=1)]

t = np.arange(0, 50, .1)
optimized_parameters = []
with open('HM-{}-{}.txt'.format(station_id, 38),'r') as myFile:
    for i, line in enumerate(myFile):
        amplitude, speed, phase  = line.split(',')
        optimized_parameters.append(float(amplitude))
        optimized_parameters.append(float(speed))
        optimized_parameters.append(float(phase))

graph2 = harmonicConstituentModel(t, *optimized_parameters)
plt.plot(data[:500,0], data[:500,1], label='Measured Currents')
plt.plot(t, graph2, label='Least Squares Fit')
plt.legend(loc='best')
plt.xlabel('Time (hours)')
plt.ylabel('Velocity (m/s)')
plt.show()


# In[ ]:



