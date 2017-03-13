import math #put 'math.' in front of all functions
# numpys and pandas?
import LatLon #latitude-longitude library

#5 to 20 kW turbines offer the best potential right now

def installation_cost(sea_cable_cost, land_cable_cost, turbine, gearbox,  ):
    """what are the arguments going to be?
    power output, lat, long, 2lat, 2long (for grid location), distance, material?,
    fuel, others, min_voltage_drop, power_loss, survey?"""
    # this function will just add up the cost of each individual function?
    # power output will help to adaquately size the wire
    #minimum voltage drop, minimum power drop (<5%), safety
    if distance <=   :
    elif distance >=   :
    elif distance >=   :




def sea_distance(lat,shore_lat,long,shore_long):
    sq1 = (lat - shore_lat)**2
    sq2 = (long - shore_long)**2
    #this calculates the distance between the turbine and the shore
    #need this distance to determine how long submarine cable will be
    return math.sqrt(sq1+sq2)

def land_distance (grid_lat,shore_lat,grid_long,shore_long):
    sq3 = (shore_lat - grid_lat)**2
    sq4 = (shore_long - grid_long)**2
    #this calculates the distance between the onshore point & load center (grid)
    #this cable does not need to be made for submarine conditions
    return math.sqrt(sq3+sq4)

def convertLatLon(self, tude):
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(tude[:-3].split('° ')))
    #this code is from Jason

def sea_cable_cost(sea_distance):
    cc1 = 0                         #cable cost 1, based on distance
        if sea_distance <= 50:
            cc1 = sea_distance * """range is (0.74-1.08M) #HVAC"""
        elif sea_distance > 50:
            cc1 = sea_distance * """range is (1.5 to 1.9M) #HVDC"""
        return cc1

def land_cable_cost (land_distance, cable_type):
    cc2 = 0                         #cable cost 2, based on distance & type
        if cable_type == "overhead"
            cc2 = land_cable_distance * """idk"""
        elif cable_type == "underground"
            cc2  = land_cable_distance * """idk"""
        elif cable_type != "overhead" or type != "underground"
            return "please enter 'overhead' or 'underground' depending on which \
             type of land electrical cables your design includes"
        return cc2
        # if 230 to 345kv cable = minimu 5MM per mile, max 15MM per mile
        # Joint Base Cape Cod, Buzzards Bay, 5 miles
        # barnstable substation connection 661 Oak St, West Barnstable, MA 02668
        # ~15 miles to the cape cod canal


#keep in mind, the longer the wire run, the more electricity is lost
#copper or aluminum - choose one
def sea_minVD (sea_distance, ): #think of arguments - distance, current type, wire type
    # Ohm's Law: volts = current (amps) * resistance (ohms)
    """A)take distance
    B)value 2 for DC or 3 for AC turbine?
    C)wire size resitance per foot from a table (based on average wire gauge)
    D)maximum amps per NEC also from a table"""
    # for AC: A*B*C*D(0.67)
    # for DC: A*B*C*D
    # is voltage drop acceptable? if not increase wire size step C

def land_minVD (land_distance):
    #most likely DC cable, copper or aluminum? choose one

def power_loss():


#need variables for equipment
#turbine_cost = have user import a number
#have user input number of turbines
#have some set discount rate for buying in bulk


if __name__ == '__main__':  #Set, don't change- here if I want to run the file script
    installation_cost()
