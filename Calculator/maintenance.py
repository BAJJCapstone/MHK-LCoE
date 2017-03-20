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

def lifetimeMonteCarlo(lifetime, emergency_list, graph = False):
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
            emergency_maintenance_cost.append(emergency_maintenance_cost[-1]+maintenance_type.event_cost)

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
    def __init__(self, total_cost):
        self.cost = total_cost*.05

class EmergencyMaintenance: #General model for an emergency maintenance
    def __init__(self, minimal_rate, midlevel_rate, severe_rate,
                minimal_cost, midlevel_cost, severe_cost, number, labor, partname = None): #maybe this should be kwargs?

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
        self.downtime = downtime = datetime.timedelta(days = 3)
        self.event_cost = self.number * self.minimal_cost + self.labor*downtime.total_seconds()/3600

    def midlevel(self):
        self.downtime = downtime = datetime.timedelta(weeks = 1)
        self.event_cost = self.number * self.midlevel_cost + self.labor*downtime.total_seconds()/3600

    def severe(self):
        self.downtime = downtime = datetime.timedelta(weeks = 2)
        self.event_cost = self.number * self.severe_cost + self.labor*downtime.total_seconds()/3600

#
# class Blade(EmergencyMaintenance):
# 
#     def minimal(self):
#         self.downtime = downtime = datetime.timedelta(days = 3)
#         self.event_cost = self.number * self.minimal_cost + self.labor*downtime
#
#     def midlevel(self):
#         self.downtime = downtime = datetime.timedelta(weeks = 1)
#         self.event_cost = self.number * self.midlevel_cost + self.labor*downtime
#
#     def severe(self):
#         self.downtime = downtime = datetime.timedelta(weeks = 2)
#         self.event_cost = self.number * self.severe_cost + self.labor*downtime
#
# class SupportColumn(EmergencyMaintenance):
#     pass
#
# class GearBox(EmergencyMaintenance):
#
#     def minimal(self):
#         self.downtime = downtime = datetime.timedelta(days = 3)
#         self.event_cost = self.number * self.minimal_cost + self.labor*downtime
#
#     def midlevel(self):
#         self.downtime = downtime = datetime.timedelta(weeks = 1)
#         self.event_cost = self.number * self.midlevel_cost + self.labor*downtime
#
#     def severe(self):
#         self.downtime = downtime = datetime.timedelta(weeks = 2)
#         self.event_cost = self.number * self.severe_cost + self.labor*downtime
#
# class Brake(EmergencyMaintenance)
#     pass
#
# class ElectricityGenerator(EmergencyMaintenance)
#     pass
#
# class Shaft(EmergencyMaintenance)
#     pass
#
# class Cable(EmergencyMaintenance)
#     pass
#
# class Maintenance:
#     # Components of the machinery
#     def __init__(self, number_of_turbines):
#         #put all of the user inputs in here
#         costs = []
#         costs.append(self.blade())
#         costs.append(self.support_column())
#         costs.append(self.gear_box())
#         costs.append(self.electricity_generator())
#         costs.append(self.shaft())
#         costs.append(self.brake())
#         costs.append(self.cable())
#         #...etc.
#
#         self.turbineCost = sum(costs)


    #def blade(self, blade_cost, number_of_blades):
    #    self.downtime = datetime.timedelta(weeks = 1, days = 3) #options here are weeks, days, hours, minutes and some other stuff
    #    return blade_cost * number_of_blades
    #def support_column(self, column_cost, number_of_columns):
    #    self.downtime = datetime.timedelta(weeks = 1, days = 3 )
    #    return column_cost * number_of_columns
    #def gear_box(self, gear_box_cost, number_of_gear_boxes):
    #    return gear_box_cost * number_of_gear_boxes
    #def electricity_generator(self, electricity_generator_cost, number_of_electricity_generators):
    #    return electricity_generator_cost * number_of_electricity_generators
    #def shaft(self, shaft_cost, number_of_shafts):
    #    return shaft_cost * number_of_shafts
    #def brake(self, brake_cost, number_of_brakes):
    #    return brake_cost * number_of_brakes
    #def cable(self, cable_cost, number_of_cables):
    #    return cable_cost * number_of_cables

    # turbine will be sum of above costs



    # Additional factors

    #def labor(self, labor_cost, number_of_laborers):
    #    return labor_cost * number_of_laborers
    #def part_life(self):
    #    return
    #def mechanical_loading(self):
    #    return
    #def weather(self):
    #    return
