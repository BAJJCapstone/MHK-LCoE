class Station:

    def __init__(self, ID):
        self.ID = ID
        self.getStationInfo()

    def convertLatLon(self, tude):
        multiplier = 1 if tude[-1] in ['N', 'E'] else -1
        return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(tude[:-3].split('° ')))

    def convertDates(self, date_listoflists, format):

        for i, dates in enumerate(date_listoflists):
            for j, date in enumerate(dates[:2]):
                if date == '':
                    currents_station_info[key][0][1] = datetime.datetime.now()
                    continue
                split_date = re.findall(r"[\w']+", date)
                if format == 'YYYY/MM/DD HH:MM':
                    date_listoflists[i][j] = datetime.datetime(year = int(split_date[0]),
                            month = int(split_date[1]),
                            day = int(split_date[2]),
                            hour = int(split_date[3]),
                            minute = int(split_date[4]))
                elif format == 'Mon DD,YYYY HH:MM':
                    date_listoflists[i][j] = datetime.datetime(year = int(split_date[2]),
                                month = int(strptime(split_date[0],'%b').tm_mon),
                                day = int(split_date[1]),
                                hour = int(split_date[3]),
                                minute = int(split_date[4])))
                else:
                    raise ValueError('Invalid Date Format Option')
