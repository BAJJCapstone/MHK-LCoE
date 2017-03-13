import datetime

class EmergencyMaintenance: #General model for an emergency maintenance
    def __init__(self, minimal_rate, midlevel_rate, severe_rate,
                minimal_cost, midlevel_cost, severe_cost, number, labor):

        self.minimal_rate = minimal_rate
        self.midlevel_rate = midlevel_rate
        self.severe_rate = severe_rate

        self.minimal_cost = minimal_cost
        self.midlevel_cost = midlevel_cost
        self.severe_cost = severe_cost

        self.number = number
        self.labor = labor

        self.downtime = None
        self.event_cost = None

    def minimal(self):
        self.downtime = downtime = datetime.timedelta(days = 3)
        self.event_cost = self.number * self.minimal_cost + self.labor*downtime

    def midlevel(self):
        self.downtime = downtime = datetime.timedelta(weeks = 1)
        self.event_cost = self.number * self.midlevel_cost + self.labor*downtime

    def severe(self):
        self.downtime = downtime = datetime.timedelta(weeks = 2)
        self.event_cost = self.number * self.severe_cost + self.labor*downtime


class Blade(EmergencyMaintenance):

    def minimal(self):
        self.downtime = downtime = datetime.timedelta(days = 3)
        self.event_cost = self.number * self.minimal_cost + self.labor*downtime

    def midlevel(self):
        self.downtime = downtime = datetime.timedelta(weeks = 1)
        self.event_cost = self.number * self.midlevel_cost + self.labor*downtime

    def severe(self):
        self.downtime = downtime = datetime.timedelta(weeks = 2)
        self.event_cost = self.number * self.severe_cost + self.labor*downtime

class SupportColumn(EmergencyMaintenance):
    pass

class GearBox(EmergencyMaintenance):

    def minimal(self):
        self.downtime = downtime = datetime.timedelta(days = 3)
        self.event_cost = self.number * self.minimal_cost + self.labor*downtime

    def midlevel(self):
        self.downtime = downtime = datetime.timedelta(weeks = 1)
        self.event_cost = self.number * self.midlevel_cost + self.labor*downtime

    def severe(self):
        self.downtime = downtime = datetime.timedelta(weeks = 2)
        self.event_cost = self.number * self.severe_cost + self.labor*downtime








class Maintenance:
    # Components of the machinery
    def __init__(self, number_of_turbines):
        #put all of the user inputs in here
        costs = []
        costs.append(self.blade())
        costs.append(self.support_column())
        costs.append(self.gear_box())
        costs.append(self.electricity_generator())
        costs.append(self.shaft())
        costs.append(self.brake())
        costs.append(self.cable())
        #...etc.

        self.turbineCost = sum(costs)


    def blade(self, blade_cost, number_of_blades):
        self.downtime = datetime.timedelta(weeks = 1, days = 3) #options here are weeks, days, hours, minutes and some other stuff
        return blade_cost * number_of_blades
    def support_column(self, column_cost, number_of_columns):
        return
    def gear_box(self, gear_box_cost, number_of_gear_boxes):
        return
    def electricity_generator(self, electricity_generator_cost, number_of_electricity_generators):
        return
    def shaft(self, shaft_cost, number_of_shafts):
        return
    def brake(self, brake_cost, number_of_brakes):
        return
    def cable(self, cable_cost, number_of_cables):
        return

    # turbine will be sum of above costs



    # Additional factors

    def labor(self, labor_cost, number_of_laborers):
        return
    def part_life(self):
        return
    def mechanical_loading(self):
        return
    def weather(self):
        return

if __name__ == '__main__':
    cost = Maintenance()


#Planned maintenance will be approx. 5% of total cost
