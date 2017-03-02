from .station import Station
from bs4 import BeautifulSoup
import urllib.request
import json
import pandas as pd
import re
import time
from time import strptime
import datetime
import numpy as np
from .retry_decorator import retry
import os.path
import matplotlib
from matplotlib import pyplot as plt

class TidalStation(Station):

    def getStationInfo(self):
        station_url = 'https://tidesandcurrents.noaa.gov/stationhome.html?id={}'.format(self.ID)
        with urllib.request.urlopen(station_url) as url:
            station_html = url.read()

        station_soup = BeautifulSoup(station_html, "html.parser")
        table = station_soup.find_all('tr') #find the table of available data
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

    def getHarmonicConstituents(self, timezone = 'local', units = 'meters'):

        harmonicConstituentDict = {}
        if timezone == 'local': timezone_option = 1
        elif timezone == 'GMT': timezone_option = 0
        else: raise ValueError('Timezone must be local or GMT')

        if units == 'meters': unit_option = 0
        elif units == 'feet': unit_option = 1
        else: raise ValueError('Units must be meters or feet')

        HM_url = 'https://tidesandcurrents.noaa.gov/harcon.html?unit={}&timezone={}&id={}'.format(unit_option, timezone_option, self.ID)
        with urllib.request.urlopen(HM_url) as url:
            HM_html = url.read()

        HM_soup = BeautifulSoup(HM_html, "html.parser")
        table_headers = [header.get_text() for header in HM_soup.find_all('th')]
        harmonicConstituents = HM_soup.find_all('tr')#find the table of available constituents
        for constituent in harmonicConstituents[1:]:
            rowData = [td.get_text() for td in constituent.find_all('td')]
            zippedData = list(zip(table_headers, rowData))
            name = rowData[1]
            harmonicConstituentDict[name] = {header: value for header, value in zippedData[2:]}

        return harmonicConstituentDict

    def graphHarmonicConstituent(self,time):
        time, height = self.predictWaterLevels(time)
        plt.plot(time, height)
        plt.show()

    def predictWaterLevels(self, time):
        if time < 7*24: increment = .1
        else: increment = 1.
        time = np.arange(0, time, increment) #one hour

        height = np.zeros_like(time)
        harmonicConstituentDict = self.getHarmonicConstituents()

        for key, constituent in harmonicConstituentDict.items():
            height += self.constituentCalculation(time, constituent)

        return time, height

    def constituentCalculation(self, time, constituent):
        return float(constituent['Amplitude'])*np.cos(np.pi/180.*(float(constituent['Speed'])*time + float(constituent['Phase'])))
