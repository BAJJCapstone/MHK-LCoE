import math #put 'math.' in front of all functions
# numpys and pandas?

class CapitalInstallation:
    def __init__(self, name, time, timePerTurbine, costPerDay, numberOfTurbines = 1, scalePerTurbine = False):
        self.timePerTurbine = timePerTurbine
        self.time = time
        self.costPerDay = costPerDay
        self.numberOfTurbines = numberOfTurbines
        self.scalePerTurbine = scalePerTurbine
        self.name = name

        self.capEx = self.calculateInstallationCost()

    def calculateInstallationCost(self):
        if self.scalePerTurbine: return self.numberOfTurbines*self.timePerTurbine*self.costPerDay
        else: return self.time*self.costPerDay

# def installation_cost(sea_cable_cost, land_cable_cost, turbine, gearbox,  ):
#     """arguments distance, material, fuel, others, min_voltage_drop, power_loss, survey?"""
#     # this function will just add up the cost of each user input + function
#     #minimum voltage drop, minimum power drop (<5%), safety
#     #variable 'voltage': user input voltage in kV
#     #variable 'circuit': user input single or double circuit for cables
#     #variable 'terrain': user input for land based cable
#     if powergen <= #
#     if powergen > #
#     #these inputs will be based on power generation model
#     if turbine_type.upper() == ""
#
#     if turbine_type.upper() == ""
#
#     if turbine_type.upper() == ""
#     #find numbers for different types of turbines, different sizes, different gear boxes
#
#
#     #find installation time frame (mainly depends on permitting, labor costs associated with that)

def convertLatLon(self, tude):
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(tude[:-3].split('° ')))
    #this code is from Jason

#need to convert lat and long to distance in miles for below
def distance(point_one, point_two):
    '''
    point_one: tuple containing (lat, lon)
    point_two: tuple containing (lat, lon)
    '''
    R = 6.371e6
    phi_one = point_one[0] * np.pi/180.
    phi_two = point_two[0] * np.pi/180.
    delta_phi = phi_two - phi_one
    delta_lamb = (point_two[1] - point_one[1])*np.pi/180
    a = np.sin(delta_phi/2.) * np.sin(delta_phi/2) + np.cos(phi_one) * np.cos(phi_two) * np.sin(delta_lamb/2) * np.sin(delta_lamb/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def sea_cable_cost(sea_distance, multiplier, multiplier_2):
    if sea_distance <= 50:
        cc1 = sea_distance * multiplier    #range is (0.74-1.08M) #HVAC"""
    elif sea_distance > 50:
        cc1 = sea_distance * multiplier_2    #range is (1.5 to 1.9M) #HVDC
    return cc1

def land_cable_cost(land_distance, cable_type, voltage, terrain, circuit = None):
    '''
    cable_type : overhead, underground
    circuit : single, double
    terrain : flat, farmland, suburban, urban, forested, hill
    '''
    cc2 = 0
    if cable_type.upper() == 'OVERHEAD':                  #cable cost 2
        if voltage <=765:   #765kV is the current maximum AC voltage in the US
            cc2 = 3969.87 * voltage         #now you have your average cost/mile of HVAC cable
        #this average assumes aluminum reinforced cable, that is greater than 10 miles
            if circuit.strip().upper() == "SINGLE":
                pass                    #cost/mile for single circuit
            elif circuit.strip().upper() == "DOUBLE":
                cc2 *= 1.6              #cost/mile for double circuit cables
            else:
                raise ValueError('Voltage under 765V needs circuit type')

        else:
            cc2 = 2880.73 * voltage     #cost/mile of HVDC aluminum overhead cable
            if circuit != None:
                raise ValueError('circuit cannot be specified for greater than 765V')

        if land_distance > 10:      #multipliers for cable distance
            pass                    #shorter the distance, more costly
        elif land_distance > 3:
            cc2 *= 1.2
        else:
            cc2 *= 1.5

        if terrain.strip().upper() == "FLAT" or "FARMLAND":     #terrain multiplier
            pass
        elif terrain.strip().upper() == "SUBURBAN":
            cc2 *= 1.27
        elif terrain.strip().upper() == "URBAN":
            cc2 *= 1.59
        elif terrain.strip().upper() =="FORESTED":
            cc2 *= 2.25
        elif terrain.strip().upper() == "HILL":
            cc2 *= 1.4

    else:
        raise TypeError('Cable Type not valid')
    return cc2

    # if cable_type.upper() == "UNDERGROUND":
    #     if land_distance <=40:      #HVAC
    #         cc2 *= 5.5 #multiplier
    #     else land_distance >40:
    #         #DC
    #         cc2  = land_cable_distance *  #cost per mile of underground cable (copper)
    #     else:
    #         raise TypeError('Cable Type not valid')
    #this cost is based on calculations from 2014 (when paper was written)
    #need to account for 2% inflation every year
    #this cost DOES include for the installation of the cable



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


if __name__ == '__main__':  #Set, don't change- here if I want to run the file script
    #need variables for equipment
    #turbine_cost = have user import a number
    #have user input number of turbines
    #have some set discount rate for buying in bulk
    installation_cost()
