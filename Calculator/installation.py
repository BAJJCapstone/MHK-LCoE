import math #put 'math.' in front of all functions
# numpys and pandas?
import LatLon #latitude-longitude library


#5 to 20 kW turbines offer the best potential right now

def installation_cost():
    """what are the arguments going to be?
    power output, lat, long, 2lat, 2long (for grid location), distance, material?,
    fuel, others, min_voltage_drop, power_loss, survey?""" #can drop lat and long arguments since distance formula
    # power output will help to adaquately size the wire
    #minimum voltage drop, minimum power drop (<5%), safety
    if distance <=   :
    elif distance >=   :
    elif distance >=   :


def distance(lat,two_lat,long,two_long):
    sq1 = (lat-two_lat)**2
    sq2 = (long-two_long)**2
    return math.sqrt(sq1+sq2)
    #this calculates the distance between the turbine and the load
    #need this distance to determine how long cable will need to be
    #how expensive the cable is based on length
    #keep in mind, the longer the wire run, the more electricity is lost

def cable_cost(distance, type, ):  #submarine cable
"""will be based on distance, cable type, cable """



def min_voltage_drop(distance, ): #think of arguments - distance, current type, wire type
    # Ohm's Law: volts = current (amps) * resistance (ohms)
    """A)take distance
    B)value 2 for DC or 3 for AC turbine?
    C)wire size resitance per foot from a table (based on average wire gauge)
    D)maximum amps per NEC also from a table"""
    # for AC: A*B*C*D(0.67)
    # for DC: A*B*C*D
    """is voltage drop acceptable? if not increase wire size step C"""

def power_loss():


if __name__ == '__main__':  #Set, don't change- here if I want to run the file script
    installation_cost()
