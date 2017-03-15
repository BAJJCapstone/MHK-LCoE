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
    #variable 'voltage': user input voltage in kV
    #variable 'circut': user input single or double circut for cables
    #variable 'terrain': user input for land based cable
    if distance <=   :
    elif distance >=   :
    elif distance >=   :




def convertLatLon(self, tude):
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(tude[:-3].split('° ')))
    #this code is from Jason

#need to convert lat and long to distance in miles for below

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

def sea_cable_cost(sea_distance):
    cc1 = 0                         #cable cost 1, based on distance
        if sea_distance <= 50:
            cc1 = sea_distance *    #range is (0.74-1.08M) #HVAC"""
        elif sea_distance > 50:
            cc1 = sea_distance *    #range is (1.5 to 1.9M) #HVDC
        return cc1

def land_cable_cost (land_distance, cable_type, voltage, circut, terrain):
    '''
    cable_type : overhead, underground
    circut : single, double
    terrain : flat, farmland, suburban, urban, forested, hill
    '''
    cc2 = 0                                          #cable cost 2
    if cable_type == "overhead" and voltage <=765:   #765kV is the current maximum AC voltage in the US
        cc2 = 3969.87 * voltage         #now you have your average cost/mile of HVAC cable
        #this average assumes aluminum reinforced cable, that is greater than 10 miles
        if circut == "single":
            pass                    #cost/mile for single circut
        else circut == "double":
            cc2 *= 1.6              #cost/mile for double circut cables

        if land_distance > 10:      #multipliers for cable distance
            pass                    #shorter the distance, more costly
        elif land_distance > 3:
            cc2 *= 1.2
        else:
            cc2 *= 1.5

        if terrain == "flat" or "farmland":     #terrain multiplier
            pass
        elif terrain == "suburban":
            cc2 *= 1.27
        elif terrain == "urban":
            cc2 *= 1.59
        elif terrain =="forested":
            cc2 *= 2.25
        elif terrain == "hill":
            cc2 *= 1.4

    elif cable_type == "overhead" and voltage > 765:
            cc2 = 2880.73 * voltage     #cost/mile of HVDC aluminum overhead cable
            if land_distance > 10:      #multipliers for cable distance
                pass                    #shorter the distance, more costly
            elif land_distance > 3:
                cc2 *= 1.2
            else:
                cc2 *= 1.5

            if terrain == "flat" or "farmland":     #terrain multiplier
                pass
            elif terrain == "suburban":
                cc2 *= 1.27
            elif terrain == "urban":
                cc2 *= 1.59
            elif terrain =="forested":
                cc2 *= 2.25
            elif terrain == "hill":
                cc2 *= 1.4
#this cost is based on calculations from 2014 (when paper was written)
#need to account for 2% inflation every year

    else cable_type == "underground":

        if land_distance <=40:
            #AC
            cc2 =
        if land_distance >40:
            #DC
            cc2  = land_cable_distance *  #cost per mile of underground cable (copper)
    else:
        raise


    return cc2




#keep in mind, the longer the wire run, the more electricity is lost
#copper or aluminum - choose one (PROABABLY COPPER)
def sea_minVD (sea_distance, ): #think of arguments -  wire type
    # Ohm's Law: volts = current (amps) * resistance (ohms)
    """A)take distance (in feet)
    B)value 2 for DC or 3 for AC turbine?
    C)wire size resitance per foot from a table (based on average wire gauge)
    D)maximum amps per NEC also from a table"""
    # for AC: A*B*C*D(0.67)
    # for DC: A*B*C*D
    # is voltage drop acceptable? if not increase wire size step C
    current = 0
    VD1 = 0
    if sea_distance <= 50:      #for DC
        current = 2
        VD1 = sea_distance * current * C * D
    elif sea_distance > 50:     #for AC
        current = 3
        VD1 = sea_distance * current * C * D * 0.67 #kilovolts
    return VD1


def land_minVD (land_distance, ):
    #most likely DC cable, copper or aluminum? choose one



def power_loss():


#need variables for equipment
#turbine_cost = have user import a number
#have user input number of turbines
#have some set discount rate for buying in bulk


if __name__ == '__main__':  #Set, don't change- here if I want to run the file script
    installation_cost()
