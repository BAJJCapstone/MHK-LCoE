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

class TidalStation(Station):

    def getStationInfo(self):
        station_url = 'https://tidesandcurrents.noaa.gov/stationhome.html?id={}'.format(station_id)
        with urllib.request.urlopen(station_url) as url:
            station_html = url.read()

        soup = BeautifulSoup(station_html, "html.parser")
        table = soup.find_all('tr') #find the table of available data
        got_latitude = False
        got_longitude = False
        try:
            for table_row in table: # skip the header
                data_type = table_row.find_all('td')[0].get_text()
                if 'Latitude' in data_type and not got_latitude:
                    latitude = table_row.find_all('td')[1].get_text()
                    got_latitude = True
                if 'Longitude' in data_type and not got_longitude:
                    longitude = table_row.find_all('td')[1].get_text()
                    got_longitude = True
                if got_longitude and got_latitude:
                    break
                else:
                    raise
        except:
            print('STATION DATA RETRIEVAL FAILURE')
            return

        self.latitude = self.convertLatLon(latitude)
        self.longitude = self.convertLatLon(longitude)

    def getAvailableData(self):
