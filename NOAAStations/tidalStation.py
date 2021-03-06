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
        '''
        this function will run when TidalStation class is called, to save several
        properties of the station specified
        ~Requires internet connection~
        '''
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
        self.constituents = self.getHarmonicConstituents()

    def getHarmonicConstituents(self, timezone = 'local', units = 'meters'):
        '''
        This function will scrape the harmonic constituents from the NOAA website
        '''
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
            # print('{0} & {1:.1f} & {2} \\\\ ' .format(name, 360./float(harmonicConstituentDict[name]['Speed']), harmonicConstituentDict[name]['Description'].split('constituent')[0]))

        return harmonicConstituentDict

    def graphHarmonicConstituent(self,time_start, time_end):
        '''
        makes the nifty graph from the harmonic constituents
        '''
        time, height = self.predictWaterLevels(time_start, time_end)
        plt.plot(time, height)
        plt.show()

    def predictWaterLevels(self, time_start, time_end):
        '''
        will build time array from requested times and return array to user
        '''
        if time_end - time_start < 7*24: increment = .1
        else: increment = 1.
        times = np.arange(time_start, time_end, increment) #one hour
        height = self.constituentCalculation(times)
        return times, height

    def constituentCalculation(self, time):
        '''
        time: float or np.array
        will calculate the height of the tide, and return an array or float based on the
        time input
        '''
        if type(time) is np.ndarray: height = np.zeros_like(time)
        else: height = 0.
        for key, constituent in self.constituents.items():
            height += float(constituent['Amplitude'])*np.cos(np.pi/180.*(float(constituent['Speed'])*time + float(constituent['Phase'])))
        return height

    def velocityFromConstituent(self, time, gravity, height):
        '''
        This is a weird calculation that I don't trust yet
        '''
        return np.sqrt(gravity/height)*self.constituentCalculation(time)
