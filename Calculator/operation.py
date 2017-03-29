class OperationsCrew:
    def __init__(self, position, size, salaryPerIndividual):
        self.position = position
        self.size = size
        self.salaryPerIndividual = salaryPerIndividual

        self.annualCost = self.calculateSalaries()
    def calculateSalaries(self):
        return self.salaryPerIndividual*self.size
