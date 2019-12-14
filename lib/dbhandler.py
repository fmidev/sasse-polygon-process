# -*- coding: utf-8 -*-
"""
Database handler
"""
import sys, re, psycopg2, logging, os, datetime
from sqlalchemy import create_engine
from sqlalchemy import exc
import numpy as np
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

        self.db_params = db
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

    def get_outages(self, starttime=None, endtime=None):
        """
        Read outages from database
        """
        sql = """
        SELECT
            \"start\",\"end\",customers,date_trunc('hour', a.start AT TIME ZONE 'Europe/Helsinki' AT TIME ZONE 'UTC') + interval '1 hour' as t, ST_AsText(the_geom) as geom
        FROM
            sasse.outages a
        WHERE
            a.type not in ('maintenance', 'planned')
        """

        if starttime is not None:
            start = starttime.strftime('%Y-%m-%d %H:%M:%S')
            sql = sql + " AND \"start\" AT TIME ZONE 'Europe/Helsinki' AT TIME ZONE 'UTC' >= '{}'".format(start)
        if endtime is not None:
            end = endtime.strftime('%Y-%m-%d %H:%M:%S')
            sql = sql + " AND \"start\" AT TIME ZONE 'Europe/Helsinki' AT TIME ZONE 'UTC' <= '{}'".format(end)

        logging.debug(sql)

        return self._query(sql)

    def get_transformers(self):
        """
        Read transformers from database
        """
        sql = """
        SELECT
            a.customers, st_astext(geom) as geom
        FROM
            sasse.transformer a
        """

        return self._query(sql)


    def update_storm_ids(self, data):
        """
        Update storm ids

        data : list
               [(id, storm_id), ...]
        """
        if len(data) < 1:
            return

        values = []
        for row in data:
            values.append((row[0], row[1]))
        values = "{}".format(values)[1:-1]


        sql = """
        UPDATE sasse.stormcell as m
        SET storm_id = c.new_id
        FROM (VALUES
             {values}
        ) as c(id, new_id)
        WHERE c.id=m.id
        """.format(values=values)

        self.execute(sql)

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
        """
        Add geometry column from wkt
        """
        df.loc[:, 'geom'] = df.loc[:, 'geom'].apply(wkt.loads)
        df = gpd.GeoDataFrame(df, geometry='geom')
        # Add centroid if necessary
        if 'centroid' not in df.columns:
            df.loc[:, 'centroid'] =  df.loc[:,'geom'].centroid

        return df

    def update_storm_objects(self, df, params):
        """
        Save storm objects to the db

        df : DataFrame
             DataFrame containing polygons and their features
        params : lst
                 List of feature names to store (used to extract features from df)
        """
        if df is None or len(df) < 1:
            return

        self.update_storm_ids(df.loc[:,['id', 'storm_id']].fillna('NULL').values)

        df.set_index('id', inplace=True)
        features = df.loc[:, params]# .replace(-99, None)
        engine = create_engine('postgresql://{user}:{passwd}@{host}:5432/{db}'.format(user=self.db_params['user'],
                                                                                     passwd=self.db_params['password'],
                                                                                     host=self.db_params['host'],
                                                                                     db=self.db_params['database']))

        table_name = 'stormcell_features'
        index_name = 'polygon_id'
        try:
            features.to_sql(table_name, engine, schema='sasse', if_exists='append', index_label=index_name)
        except exc.SQLAlchemyError:
            logging.warning('Rows already exist, not updating')

        with engine.connect() as con:
            try:
                con.execute('ALTER TABLE sasse.{} ADD PRIMARY KEY ({});'.format(table_name, index_name))
            except exc.SQLAlchemyError:
                pass

    def update_classification_dataset(self, df, table_name='classification_dataset'):
        """
        Save classification dataset into the db

        df : DataFrame
             DataFrame data
        """
        if df is None or len(df) < 1:
            return
        logging.info('Storing classification set to db sasse.{}...'.format(table_name))
        #df.set_index('id', inplace=True)
        engine = create_engine('postgresql://{user}:{passwd}@{host}:5432/{db}'.format(user=self.db_params['user'],
                                                                                     passwd=self.db_params['password'],
                                                                                     host=self.db_params['host'],
                                                                                     db=self.db_params['database']))

        index_name = 'id'
        try:
            df.to_sql(table_name, engine, schema='sasse', if_exists='append', index=False)
        except exc.SQLAlchemyError:
            logging.warning('Rows already exist, not updating')
