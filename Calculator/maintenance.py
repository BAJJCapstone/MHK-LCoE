import datetime
import random
import numpy as np

import matplotlib.pyplot as plt

def monteCarlo(emergency_list):
    '''
    Accepts list of EmergencyMaintenance objects to determine the next event time and
    event.
    Returns the type of maintenance that is occuring and the amount of time until the
    next EmergencyMaintenance event (in years).
    '''
    sum_rates = 0
    events = []
    first_loop = True
    for emergency in emergency_list:
        if first_loop:
            events.append(float(emergency.minimal_rate))
            first_loop = False
        else:
            events.append(events[-1]+emergency.minimal_rate)
        events.append(events[-1]+emergency.midlevel_rate)
        events.append(events[-1]+emergency.severe_rate)
        sum_rates += emergency.minimal_rate + emergency.midlevel_rate + emergency.severe_rate

    wait_time = random.expovariate(sum_rates)
    choice = random.uniform(0, sum_rates)

    for i, event in enumerate(events):
        if event > choice:
            break

    severity_level = i % 3
    maintenance_type = emergency_list[i//3]

    if severity_level == 0:
        maintenance_type.minimal()
    elif severity_level == 1:
        maintenance_type.midlevel()
    else:
        maintenance_type.severe()

    return maintenance_type, wait_time

def lifetimeMonteCarlo(lifetime, emergency_list, discount_rate = .05, graph = False):
    '''
    lifetime: argument in number of years to run simulation
    graph: boolean to graph results or not
    *args: list of parts included in emergency maintenances
    '''
    if graph:
        time = [0]
        emergency_maintenance_cost = [0]
        while 1:
            maintenance_type, wait_time = monteCarlo(emergency_list)
            current_time = time[-1] + wait_time + maintenance_type.downtime.total_seconds()/(24*3600*365.25)
            if current_time > lifetime: break
            time.append(current_time)
            emergency_maintenance_cost.append((emergency_maintenance_cost[-1]+maintenance_type.event_cost)/((1+discount_rate)**(current_time)))

        plt.plot(time, emergency_maintenance_cost)
        plt.ylabel('Cost (US $)')
        plt.xlabel('Time (years)')
        plt.show()

    else:
        time = 0
        emergency_maintenance_cost = 0
        while 1:
            maintenance_type, wait_time = MonteCarlo(emergency_list)
            current_time = time[-1] + wait_time + maintenance_type.downtime
            if current_time > lifetime: break
            time += wait_time + maintenance_type.downtime
            emergency_maintenance_cost += maintenance_type.event_cost

    return time, emergency_maintenance_cost

class PlannedMaintenance:
    '''
    total_cost : this will just take a chunk of the total capital cost and apply it the the planned maintenance by industry conventions
    '''
    def __init__(self, total_cost):
        self.cost = total_cost*.05

class EmergencyMaintenance: #General model for an emergency maintenance
    '''
    General Maintenance class
    minimal_rate: yearly rate of failure
    midlevel_rate: yearly rate of failure
    severe_rate: yearly rate of failure
    minimal_cost: US$ capital associated with minimal failure
    midlevel_cost: US$ capital associated with midlevel failure
    severe_cost: US$ capital associated with severe failure
    number: how many need to be replaced
    labor: cost of labor/hr
    minimal_dt: downtime in days for minimal event
    midlevel_dt: downtime in days for midlevel event
    severe_dt: downtime in days for severe event
    '''
    def __init__(self, minimal_rate, midlevel_rate, severe_rate,
                minimal_cost, midlevel_cost, severe_cost, number=1 , labor=40,
                minimal_dt = 3, midlevel_dt = 7, severe_dt = 14, partname = None): #maybe this should be kwargs?

        self.minimal_rate = minimal_rate
        self.midlevel_rate = midlevel_rate
        self.severe_rate = severe_rate

        self.minimal_cost = minimal_cost
        self.midlevel_cost = midlevel_cost
        self.severe_cost = severe_cost

        self.number = number
        self.labor = labor

        self.partname = partname
        self.downtime = None
        self.event_cost = None

    def minimal(self):
        self.downtime = downtime = datetime.timedelta(days = minimal_dt)
        self.event_cost = self.number * self.minimal_cost + self.labor*downtime.total_seconds()/3600

    def midlevel(self):
        self.downtime = downtime = datetime.timedelta(days = midlevel_dt)
        self.event_cost = self.number * self.midlevel_cost + self.labor*downtime.total_seconds()/3600

    def severe(self):
        self.downtime = downtime = datetime.timedelta(days = severe_dt)
        self.event_cost = self.number * self.severe_cost + self.labor*downtime.total_seconds()/3600
