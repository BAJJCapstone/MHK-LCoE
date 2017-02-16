class Station:

    def __init__(self, ID):
        self.ID = ID
        self.getStationInfo()

    def convertLatLon(self, tude):
        multiplier = 1 if tude[-1] in ['N', 'E'] else -1
        return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(tude[:-3].split('° ')))
