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


    def blade(self):
        return 
    def support_column(self):
        return
    def gear_box(self):
        return
    def electricity_generator(self):
        return
    def shaft(self):
        return
    def brake(self):
        return
    def cable(self):
        return

    # turbine will be sum of above costs



    # Additional factors

    def labor(self):
        return
    def part_life(self):
        return
    def mechanical_loading(self):
        return
    def weather(self):
        return

if __name__ == '__main__':
    cost = Maintenance()
