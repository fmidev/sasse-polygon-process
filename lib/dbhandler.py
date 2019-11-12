# -*- coding: utf-8 -*-
"""
Database handler
"""
import sys
import re
import psycopg2
import logging
import os
import numpy as np
import datetime
from collections import defaultdict
from configparser import ConfigParser
import pandas.io.sql as sqlio
import geopandas as gpd
from shapely import wkt

class DBHandler(object):
    """
    Handle database connection and provide
    methods to insert and read storm objects to database
    """
    return_df = True

    def __init__(self, config_filename, config_name):
        self.config_filename = config_filename
        self.config_name = config_name
        self._connect()

    def _connect(self):
        """ Create connection if needed """
        params = self._config(self.config_filename, self.config_name)
        self.conn = psycopg2.connect(**params)
        return self.conn

    def _config(self, config_filename, section=''):
        parser = ConfigParser()
        parser.read(config_filename)
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, self.config_filename))

        return db


    def get_polygons(self, filters):
        """
        Get polygons

        filters : dict
                  filters {'name': []}

        return lst
        """
        sql = """
        SELECT id,storm_id,point_in_time,weather_parameter,low_limit,high_limit,ST_AsText(geom) as geom
        FROM sasse.stormcell
        WHERE 1=1"""

        if 'time' in filters:
            sql += " AND point_in_time IN ('{}')".format(','.join(filters['time']))

        return self._query(sql)

    def execute(self, statement):
        """
        Execute single SQL statement in
        a proper manner
        """

        self._connect()
        with self.conn as conn:
            with conn.cursor() as curs:
                curs.execute(statement)

    def _query(self, sql):
        """
        Execute query and return results

        sql str sql to execute

        return list of sets
        """
        self._connect()
        with self.conn as conn:
            if self.return_df:
                return self._df_to_geodf(sqlio.read_sql_query(sql, conn))
            else:
                with conn.cursor() as curs:
                    curs.execute(sql)
                    results = curs.fetchall()
                    return results

    def _df_to_geodf(self, df):
        """ Add geometry column from wkt """
        df.loc[:, 'geom'] = df.loc[:, 'geom'].apply(wkt.loads)
        df = gpd.GeoDataFrame(df, geometry='geom')
        # Add centroid if necessary
        if 'centroid' not in df.columns:
            df.loc[:, 'centroid'] =  df.loc[:,'geom'].centroid

        return df
