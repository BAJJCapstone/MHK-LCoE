
# coding: utf-8

# In[4]:

from NOAAStations import TidalStation
from DeviceModels import Turbine, calculate_power
from Calculator import maintenance, operation, installation

import pandas as pd
import os
from ipywidgets import widgets, interact, fixed
from IPython.display import display
get_ipython().magic('matplotlib inline')
import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
import scipy
import scipy.interpolate
from contextlib import redirect_stdout
figsize(12, 10)
sbn.set_context("paper", font_scale=1)
sbn.set_style("whitegrid")


from collections import namedtuple


# ### Testing for the maintenance monte carlo simulation

# In[5]:

def harmonicConstituentModel(time, *hm):
    assert len(hm) % 3 == 0
    velocity = 0 
    for i in range(len(hm)//3):
        velocity += hm[3*i]*np.cos((hm[3*i+1] * time + hm[3*i+2])*np.pi/180.)
    return velocity

def calculate_Installation(installations):
    return sum([i.capEx for i in installations])

def calculate_ops(ops, time):
    return(sum([o.annualCost*time for o in ops]))

def richardsCurve(Velocity,B,M,g):
    return 1200*(1+.1835*np.exp(-1*B*(Velocity-M)))**(-1/g)

def interpolate(value, from_a, from_b, to_a, to_b):
    return to_a +((to_a-to_b)/(from_a-from_b))*(value - from_a)        

def LevelizedCostofElectricity(HM,
                               number_of_turbines,
                               lifetime, 
                               K, Q, B, M, g,
                               emergency_maintentance,
                               installation,
                               operations,
                               power_array = None,
                               num = 500):
    
    '''
    This function will calculated the levelized cost of electricity given the parameters for maintenance, power generation, installation
    and lifetime
    station_id will determine the location due to the necessity to use harmonic constituents for the calculations
    grid_location is where the connections will be made
    cap_ex are the capital expenditures for the cost of the turbine and fixtures
    this function was written with a sensitivity analysis in mind
    '''
    
    MCT = Turbine(K, Q, B, M, g)
        
    if power_array is None:
        power_array , time_array = calculate_power(HM, 
                                                     MCT, 
                                                     0, 
                                                     0, 
                                                     lifetime*24*3600*365.25, 
                                                     ) # everything else is in years, need to convert to seconds for int
        time_array = time_array/(24.*3600.*365.25)
    else: 
        time_array = np.linspace(0, lifetime, len(power_array))
    ###
    # The following code is used to run the monte carlo simulation with feedback to the power generation functions
    # where the downtime requires the turbine to no longer generate an output
    ###

    power_array *= .95 #to account for voltage drop across cable
    maintenance_costs = np.zeros(num)
    power_losses = np.zeros_like(maintenance_costs)
    #time to run the simulation
    
    for i in range(num):            
        end_loop = False
        time_tracker = 0.
        power_loss = 0.
        maintenance_cost = 0.
        
        for turbine in range(number_of_turbines):
            while not end_loop:
                maintenance_event, uptime = maintenance.monteCarlo(emergency_maintentance)
                end_time = time_tracker + uptime
                maintenance_cost += maintenance_event.event_cost
                time_tracker += uptime + maintenance_event.downtime.total_seconds()/(24*3600*365.25)
                if end_time >= lifetime or time_tracker >= lifetime:
                    break
                start_index = np.searchsorted(time_array, time_tracker)
                end_index = np.searchsorted(time_array, end_time)
                energy_2 = interpolate(time_tracker, time_array[start_index-1], time_array[start_index], 
                                       power_array[start_index-1], power_array[start_index])
                energy_1 = interpolate(end_time, time_array[end_index-1], time_array[end_index], 
                                       power_array[end_index-1], power_array[end_index])
                power_loss += energy_2 - energy_1

        power_losses[i] = power_loss
        maintenance_costs[i] = maintenance_cost

    installation_cost = calculate_Installation(installation)
    planned_maintenance = .05 * installation_cost
    ops_cost = calculate_ops(operations, lifetime)
    # Process the final costs and return the levelized cost
    total_cost = np.mean(maintenance_costs) + installation_cost + ops_cost + planned_maintenance
    total_power = (power_array[-1]*number_of_turbines - np.mean(power_losses))/3600 #to kWhr!!
    print('Ideal power output = {} MWhr '.format(power_array[-1]/(1000*3600)))
    print('Estimated total power loss - {} MJ, sigma = {}'.format(np.mean(power_losses)/1000, np.std(power_losses)/1000))
    print('Estimated total maintenance cost - $ {}, sigma = $ {}'.format(np.mean(maintenance_costs), np.std(maintenance_costs)))
    print('Estimated installation cost - $ {}'.format(installation_cost))
    print('Estimated operations cost - $ {}'.format(ops_cost))
    return total_cost/total_power, power_array


# In[6]:

Maintenance_Rate = namedtuple('Parameter', 'partname minimal_rate midlevel_rate severe_rate minimal_cost midlevel_cost severe_cost number labor')
CapitalInstallation = namedtuple('Parameter', 'name costPerDay timePerTurbine time numberOfTurbines scalePerTurbine')

## Heli Rate 2000-5000
## Heli Speed = 130-145
## 1014
## 31


emergency_maintenance = [
    Maintenance_Rate('Blade', 0.042, 0.0273, 0.00007, 1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.),
    Maintenance_Rate('Others', 0.03, 0.0299, 0.00006, 1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.),
    Maintenance_Rate('Gear Box',0.2125, 0.0325, 0.0005, 1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.),
    Maintenance_Rate('Electricity Generator', 0.065, 0.0545, 0.0065, 1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.),
    Maintenance_Rate('Shaft', 0.002, 0.007, .001, 1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.),
    Maintenance_Rate('Brake', 0.0153, 0.0325, 0.0025,1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.),
    Maintenance_Rate('Cable', 0.225, 0.09247, 0.000002,1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.),
    Maintenance_Rate('Control system', 0.1, 0.1, 0.0001,1014.*24.*1, 1014*24.*4, 3500*24.*5, 1., 3*40.)
]


emergency_events = [maintenance.EmergencyMaintenance(
            e.minimal_rate, 
            e.midlevel_rate, 
            e.severe_rate,
            e.minimal_cost, 
            e.midlevel_cost, 
            e.severe_cost,
            number = e.number, 
            labor = e.labor, 
            partname = e.partname)
            for e in emergency_maintenance]

ops = [
    operation.OperationsCrew('Site Manager', 1, 114750),
    operation.OperationsCrew('Admin Asst', 2, 94500),
    operation.OperationsCrew('Sr. Tech', 3, 126360/3),
    operation.OperationsCrew('Jr. Tech', 6, 219024/6),
]

HM = []
with open('HM-COI0301-38.txt','r') as myFile:
    for line in myFile:
        amplitude, speed, phase  = line.split(',')
        HM.append(float(amplitude))
        HM.append(float(speed))
        HM.append(float(phase))

HM = tuple(HM)

# MCT = Turbine(1200., 0.1835, 3.55361367,  2.30706792,  1.05659521)
# Sagamore = TidalStation(8447173)
# results, times = calculate_power(Sagamore, MCT, 0, 0, 365*24*3600, 9.8, 3)
# t = np.arange(0, 50)
# graph2 = harmonicConstituentModel(t, *HM)

# plt.plot(t, graph2, label='Least Squares Fit')
# plt.legend(loc='best')
# plt.xlabel('Time (hours)')
# plt.ylabel('Velocity (m/s)')
# plt.savefig('TidalCurrentHM.png', format='png', transparent=True, bbox_inches='tight')


# In[7]:

lifetime = 20.
LCOE = []
for number_of_turbines in [1,2,5,10,50,100]:
    print('Starting calculations for {} turbines'.format(number_of_turbines))

    Capital_Installations = [
    CapitalInstallation('Pile Installation, Mobilize Vessel', 111000., 'n/a', 4, number_of_turbines, False),
    CapitalInstallation('Pile Installation, Transport', 167000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Pile Installation, Drive Piles', 164000., .3, 'n/a', number_of_turbines, True),
    CapitalInstallation('Pile Installation, transport home', 167000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Pile Installation, Demobilize', 110000., 'n/a', 3, number_of_turbines, False),
    CapitalInstallation('Gunderboom Sound Barrier', 4500000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Frame for Barrier',50000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Mob/Demob Sound Barrier', 70000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Cable transport to site',45000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Cables install to device',75000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Cable to pile',75000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Cable Splicing',75000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Cable Fairleading',75000., 'n/a', 5, number_of_turbines, False),
    CapitalInstallation('Cable through HDD', 75000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Cable Burial', 75000., 'n/a', 4, number_of_turbines, False),
    CapitalInstallation('Cable Testing and Commissioning',63000., 'n/a', 4, number_of_turbines, False),
    CapitalInstallation('Cable Transport Home', 45000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Cable - Demobilization', 46000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Device Installation, Mobilize Vessel', 74000., 'n/a', 4, number_of_turbines, False),
    CapitalInstallation('Device Installation, Transport to site', 79000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Device Installation, install',  106000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Device Installation, Secure Cables', 106000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Device Installation, Fairleading Cables',  106000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Device Installation, Transport Home', 87000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('FERC Filing Fee', 91000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Device', 3000000., 1, 'n/a', number_of_turbines, True)]

    installations = [installation.CapitalInstallation(i.name, 
                                                      i.time, 
                                                      i.timePerTurbine, 
                                                      i.costPerDay, 
                                                      i.numberOfTurbines, 
                                                      i.scalePerTurbine)
                                                      for i in Capital_Installations ]
    
    ops.append(operation.OperationsCrew('Lease', 1, 839*number_of_turbines/3))
    
    if number_of_turbines == 1:
        result, power_array = LevelizedCostofElectricity(HM, number_of_turbines, lifetime, 1200., 0.1835, 3.55361367,  2.30706792,  1.05659521,
                               emergency_events, installations, ops)
    else:
        result, power_array = LevelizedCostofElectricity(HM, number_of_turbines, lifetime, 1200., 0.1835, 3.55361367,  2.30706792,  1.05659521,
                               emergency_events, installations, ops, power_array = power_array)        
    
    LCOE.append(result)
    print('LCOE for {} turbine(s) was {}'.format(number_of_turbines, LCOE[-1]))
    print('-'*80)

print('1. SeaGen - {}'.format(LCOE))




# In[8]:

lifetime = 20.
LCOE_gen4wave = []
for number_of_turbines in [1,2,5,10,50,100]:
    print('Starting calculations for {} turbines'.format(number_of_turbines))

    Capital_Installations = [
    CapitalInstallation('Pile Installation, Mobilize Vessel', 111000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Pile Installation, Transport', 167000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Pile Installation, Drive Piles', 164000., .3, 'n/a', number_of_turbines, True),
    CapitalInstallation('Pile Installation, transport home', 167000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Pile Installation, Demobilize', 110000., 'n/a', 3, number_of_turbines, False),
    CapitalInstallation('Gunderboom Sound Barrier', 4500000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Frame for Barrier',50000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Mob/Demob Sound Barrier', 70000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Cable transport to site',45000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Cables install to device',75000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Cable to pile',75000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Cable Splicing',75000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Cable Fairleading',75000., 'n/a', 5, number_of_turbines, False),
    CapitalInstallation('Cable through HDD', 75000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Cable Burial', 75000., 'n/a', 4, number_of_turbines, False),
    CapitalInstallation('Cable Testing and Commissioning',63000., 'n/a', 4, number_of_turbines, False),
    CapitalInstallation('Cable Transport Home', 45000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Cable - Demobilization', 46000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Device Installation, Mobilize Vessel', 74000., 'n/a', 4, number_of_turbines, False),
    CapitalInstallation('Device Installation, Transport to site', 79000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Device Installation, install',  106000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Device Installation, Secure Cables', 106000., .5, 'n/a', number_of_turbines, True),
    CapitalInstallation('Device Installation, Fairleading Cables',  106000., 'n/a', 2, number_of_turbines, False),
    CapitalInstallation('Device Installation, Transport Home', 87000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('FERC Filing Fee', 91000., 'n/a', 1, number_of_turbines, False),
    CapitalInstallation('Device', 3000000., 1, 'n/a', number_of_turbines, True),
    CapitalInstallation('Additional Device Fee', 9000000., 'n/a', 1, number_of_turbines, False)]

    installations = [installation.CapitalInstallation(i.name, 
                                                      i.time, 
                                                      i.timePerTurbine, 
                                                      i.costPerDay, 
                                                      i.numberOfTurbines, 
                                                      i.scalePerTurbine)
                                                      for i in Capital_Installations ]
    if number_of_turbines == 1:
        result, power_array = LevelizedCostofElectricity(HM, number_of_turbines, lifetime, 1164.785 , 2.834 , 2.778 , 1.020 , 0.751,
                               emergency_events, installations, ops)
    else:
        result, power_array = LevelizedCostofElectricity(HM, number_of_turbines, lifetime, 1164.785 , 2.834 , 2.778 , 1.020 , 0.751, 
                               emergency_events, installations, ops, power_array=power_array)      
    LCOE_gen4wave.append(result)
    
    print('LCOE for {} turbine(s) was {}'.format(number_of_turbines, LCOE_gen4wave[-1]))
print('2. Gen4Wave V7 - {}'.format(LCOE_gen4wave))


# In[11]:

lifetime = 20.
LCOE_RM4 = []
for number_of_turbines in [1,2,5,10,50,100]:
    print('Starting calculations for {} turbines'.format(number_of_turbines))
    def calculateCapital(num_turbs):
        return -30482*num_turbs**2 + 3e7*num_turbs +9e7
    Capital_Installations = [
    CapitalInstallation('Generalized', calculateCapital(number_of_turbines), 'n/a', 1, number_of_turbines, False)]

    installations = [installation.CapitalInstallation(i.name, 
                                                      i.time, 
                                                      i.timePerTurbine, 
                                                      i.costPerDay, 
                                                      i.numberOfTurbines, 
                                                      i.scalePerTurbine)
                                                      for i in Capital_Installations ]
   

    ops.append(operation.OperationsCrew('Lease', 1, 839*number_of_turbines*28800/100))


    if number_of_turbines == 1:
        result, power_array = (LevelizedCostofElectricity(HM, number_of_turbines, lifetime, 3815.76, 0.83, 4.38412328, 1.32480294, 0.952668935,
                               emergency_events, installations, ops))
    else:
        result, power_array = LevelizedCostofElectricity(HM, number_of_turbines, lifetime, 3815.76, 0.83, 4.38412328, 1.32480294, 0.952668935,
                               emergency_events, installations, ops, power_array = power_array)       
    LCOE_RM4.append(result)
    
    print('LCOE for {} turbine(s) was {}'.format(number_of_turbines, LCOE_RM4[-1]))
print('3. RM4, Moored Glider, 4 axial-flow - {}'.format(LCOE_RM4))


# In[10]:

import plotly.plotly as py
import plotly.graph_objs as go

trace = [go.Bar(
    x = ['One','Two','Five','Ten','Fifty','One Hundred'],
    y = LCOE,
    marker = dict(color = '11C3F4')),
    go.Bar(
    x = ['One','Two','Five','Ten','Fifty','One Hundred'],
    y = LCOE_gen4wave,
    marker = dict(color = 'FFB450')),
    go.Bar(
    x = ['One','Two','Five','Ten','Fifty','One Hundred'],
    y = LCOE_RM4,
    marker = dict(color = '00CCCC'))]

layout = go.Layout(
    xaxis = dict(title = 'Number of SeaGen Turbines',
        titlefont = dict(
        size = 20,
        color = 'white'),
        tickfont=dict(
            size=16,
            color='white'
        )),
    yaxis = dict(title = 'US $ / kWhr',
        titlefont = dict(
        size = 20,
        color = 'white'),
        tickfont=dict(
            size=16,
            color='white'
        )),
        annotations=[
        dict(x=xi,y=yi,
             text='${0:.2f}'.format(yi[0]),
             font = dict(
                 size = 16,
                 color='white'),
             xanchor='center',
             yanchor='bottom',
             showarrow=False,
        ) for xi, yi in zip(['One','Two','Five','Ten','Fifty','One Hundred'], LCOE_gen4wave)],
    paper_bgcolor='transparent',
    plot_bgcolor='transparent')

fig = go.Figure(data = trace, layout=layout)
py.iplot(fig, filename='KnikArm')

