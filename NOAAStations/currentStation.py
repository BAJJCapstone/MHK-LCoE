from station import Station

from bs4 import BeautifulSoup
import urllib.request
import json
import pandas as pd

import re
import time
from time import strptime
import datetime

from retry_decorator import retry

import os.path

class CurrentStation(Station):


    def getStationInfo(self):
        '''
        Function is used to determine the location, sample intervals, and deployment dates of a Currents
        station, assigns as attributes

        deploymentDates = list of lists [[begin_1, end_1],[begin_2, end_2]], where length is dependent on number of
            deployments, (where dates are datetime objects?)

        latitude = latitude (as latlon object?)
        longitude = longitude (as latlon object?)

        sampleInterval = int, in minutes

        '''

        self.station_prefix = "https://tidesandcurrents.noaa.gov/cdata/StationInfo?id="

        with urllib.request.urlopen(self.station_prefix + self.ID) as url:
            current_html = url.read()

        soup = BeautifulSoup(current_html, "html.parser")
        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            headers = table.find_all('thead')
            for header in headers:
                header_titles = [th.get_text() for th in header.find_all('th')]
                singleDeploymentTitles = ['Attribute', 'Value']
                multipleDeploymentTitles = ['Deployed', 'Recovered', 'Latitude', 'Longitude']
                if all(title in singleDeploymentTitles for title in header_titles):
                    rows = table.tbody.find_all('tr')
                    dataList = [[td.get_text().strip() for td in tr.findAll("td")] for tr in rows]
                    dataDict = {data[0]: data[1] for data in dataList}
                    deploymentDates = [dataDict['Deployment/Recovery Dates (UTC)'].split(' / ')]
                elif all(title in multipleDeploymentTitles for title in header_titles):
                    rows = table.tbody.find_all('tr')
                    deploymentDates = [[td.get_text() for td in tr.findAll("td")] for tr in rows]

        self.deploymentDates = deploymentDates
        self.latitude = dataDict['Latitude']
        self.longitude = dataDict['Longitude']
        self.sampleInterval = int(dataDict['Sample Interval'].split()[0])

    def getAvailableData(self, date_list = None):
        '''
        Will go through list of lists that are assigned to station_id, requesting for data between the available dates
        '''
        if date_list == None:
            date_list = self.deploymentDates

        station_id = self.ID

        lifetime_data = []
        for date_list in date_lists:
            begin_date = date_list[0]
            begin_date += datetime.timedelta(minutes=begin_date.minute % 6)

            end_date = date_list[1]
            end_date -= datetime.timedelta(minutes=end_date.minute % 6)

            date = begin_date
            month = datetime.timedelta(days=31)
            first_loop = True
            while 1:
                end_loop = False
                next_date = date + month

                if next_date > end_date:
                    next_date = end_date
                    end_loop = True

                url = 'https://tidesandcurrents.noaa.gov/api/datagetter?'
                params = {
                    'begin_date': '{:02d}/{:02d}/{} {:02d}:{:02d}'.format(date.month, date.day, date.year, date.hour, date.minute),
                    'end_date':'{:02d}/{:02d}/{} {:02d}:{:02d}'.format(next_date.month, next_date.day, next_date.year, next_date.hour, next_date.minute),
                    'station':station_id,
                    'product':'currents',
                    'units':'metric',
                    'time_zone':'lst_ldt',
                    'application':'web_services',
                    'format':'json'
                }

                bin_list = []

                i=1
                while 1:
                    # Currents has specific request parameter for bins - I believe this are associated with depth
                    params['bin'] = i
                    resp = requests.get(url=url, params=params)
                    try:
                        bin_list.append(pd.DataFrame(resp.json()['data']))
                    except:
                        break
                    bin_list[i-1].drop('b', axis=1, inplace=True) #just we just want, direction/speed
                    bin_list[i-1].set_index('t', inplace=True) #index based off time
                    bin_list[i-1].rename(columns = lambda x : '{}.{}.{}'.format(station_id, i, x), inplace = True)
                    i += 1


                try:
                    monthly_data = pd.concat(bin_list, axis=1)
                    lifetime_data.append(monthly_data) #easiest way to combine lots of data
                except ValueError:
                    print('No available data for {}  -  {}'.format(date, next_date))
                    pass

                date = next_date
                if end_loop:
                    break

        # # Done with getting data - how do we wanna save?

        # print('Done with {}'.format(station_id))
        # try: #combine into one big dataframe
        #     lifetime_dataframe = pd.concat(lifetime_data)
        #     lifetime_dataframe.to_pickle(os.path.join(saving_directory, '{}.pkl'.format(station_id)))
        #     return lifetime_dataframe, True
        #
        # except ValueError:
        #     print('Error: No available data from - {}'.format(station_id))
        #     return None, False
