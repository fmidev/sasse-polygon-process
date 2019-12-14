# -*- coding: utf-8 -*-
"""
SmartMet Server data loader
"""
import sys, os, logging, datetime, yaml, requests, re
import numpy as np
from configparser import ConfigParser
import pandas as pd
import geopandas as gpd
from shapely import wkt

class SmartMetException(Exception):
   """ SmartMet request fails"""
   pass

class SmartMetHandler(object):
    """
    Class to load data from SmartMet Server
    """

    def __init__(self, config_filename, config_name):
        self.config_filename = config_filename
        self.config_name = config_name

        params = self._config(self.config_filename, self.config_name)

    def _config(self, config_filename, config_name):
        # Read a yaml configuration file from disk
        with open(config_filename) as conf_file:
            config_dict = yaml.safe_load(conf_file)

        self.config = config_dict[config_name]
        self.params = config_dict['params']

        return config_dict

    def get_data(self, wkt, time):
        """ Read data for given wkt and time """

        paramlist = ["FF-MS:ERA5:26:6:10:0"]
        duplicates =["Wind Speed", "Wind Direction"]

        paramlist = self.params_to_list()
        url = "{host}/timeseries?format=json&starttime={time}&endtime={time}&tz=utc&param={params}&wkt={wkt}".format(host=self.config['host'],params=','.join(paramlist), wkt=wkt.simplify(0.05, preserve_topology=True), time=time.strftime("%Y%m%dT%H%M%S"))

        logging.debug(url)

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()[0]
        else:
            raise SmartMetException({'url': url, 'response_headers': response.headers})

        met_params = {}
        for param, value in data.items():
            f = re.search('(?<=@).*(?={)', param).group()
            p = re.search(r'(?<={).*(?=})', param).group()

            # Virtual wind params and real wind params may be missing and are
            # substitute to each other depending on data
            if self.params[p]['name'] in duplicates and value is None:
                continue
            else:
                met_params[f+' '+self.params[p]['name']] = value
                
        return met_params

    def params_to_list(self, shortnames=False):
        """ Return list of queryable params """
        lst = []
        for param, info in self.params.items():
            for f in info['aggregation']:
                if shortnames:
                    lst.append(f[1:]+' '+info['name'])
                else:
                    lst.append(f+'{'+param+'}')
        return lst
