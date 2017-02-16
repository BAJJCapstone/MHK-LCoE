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


        self.latitude =
        self.longitude =

    def getAvailableData(self):
