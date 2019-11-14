# -*- coding: utf-8 -*-
"""
CSV file handler
"""
import sys, lopgging, os
#import numpy as np
#import datetime
#from configparser import ConfigParser
#import pandas.io.sql as sqlio
import pandas as pd
import geopandas as gpd
from shapely import wkt

class FileHandler(object):
    """
    Create, store and read csv files
    """

    def __init__(self, config_filename, config_name):
        self.config_filename = config_filename
        self.config_name = config_name

    
