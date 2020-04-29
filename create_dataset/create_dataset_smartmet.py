# -*- coding: utf-8 -*-
import sys, argparse, logging, joblib, requests, re
sys.path.insert(0, 'lib/')
import datetime, time, boto3, yaml, os
from pathlib import Path

from datetime import timedelta

from configparser import ConfigParser

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt

#from shapely.geometry import Polygon, MultiPolygon

import psycopg2
from sqlalchemy import create_engine
import pandas.io.sql as sqlio

from dask.distributed import Client, progress, wait
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import dask
from dask import delayed
import dask.array as dask_array

from tqdm import tqdm

from config import read_options
#from smartmethandler import SmartMetHandler

class SmartMetException(Exception):
   """ SmartMet request fails"""
   pass

def db_config(config_filename, section=''):
    parser = ConfigParser()

    my_file = Path(config_filename)
    if not my_file.is_file():
        raise Exception("{} don't exist".format(config_filename))

    parser.read(config_filename)
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, config_filename))

    return db

def create_dataset_loiste_jse(db_params, start, end, meta_params, geom_params, storm_params, outage_params, transformer_params, all_params):
    """ Gather dataset from db """
    #print('Reading data for {}-{}...'.format(start, end))

    conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (db_params['database'], db_params['user'], db_params['host'], db_params['password']))

    sql = """
        SELECT
        """
    first = True
    for p in meta_params:
        if not first:
            sql += ','
        first = False
        sql += "a.{}".format(p)

    for p in geom_params:
        sql += ',ST_AsText(a.{}) as {}'.format(p, p)

    for p in storm_params:
            sql += ',"{}"'.format(p)
    for p in outage_params:
        sql += ',c.{}'.format(p)
    for p in transformer_params:
        sql += ',d.{}'.format(p)

    sql += """
        FROM
         sasse.stormcell a
         INNER JOIN sasse.stormcell_features b ON a.id = b.polygon_id
         LEFT JOIN (
                  SELECT
                           b.id,
                           COUNT(1) AS outages,
                           SUM(customers) AS customers
                  FROM
                           sasse.outages a,
                           sasse.stormcell b
                  WHERE
                           date_trunc('hour', a.start AT TIME ZONE 'Europe/Helsinki' AT TIME ZONE 'UTC') + interval '1 hour' = point_in_time
                           AND ST_Intersects(st_setsrid (the_geom, 4326), st_setsrid (geom, 4326))
                           AND a.type NOT IN ('maintenance', 'planned')
                           AND point_in_time >= '{start}'
                           AND point_in_time <= '{end}'
                  GROUP BY
                           b.id) c ON c.id = a.id
         LEFT JOIN (
                  SELECT
                           b.id,
                           COUNT(1) AS transformers,
                           SUM(customers) as all_customers
                  FROM
                           sasse.transformer a,
                           sasse.stormcell b
                  WHERE
                           ST_Intersects(st_setsrid (a.geom, 4326), st_setsrid (b.geom, 4326))
                           AND point_in_time >= '{start}'
                           AND point_in_time <= '{end}'
                  GROUP BY
                           b.id) d ON d.id = a.id
        WHERE (st_intersects(ST_MakeEnvelope(25.5, 61.0, 29.6, 62.5, 4326), st_setsrid (geom, 4326))
            OR st_intersects(ST_MakeEnvelope(26.1, 63.7, 30.3, 65.5, 4326), st_setsrid (geom, 4326)))
        AND point_in_time >= '{start}'
        AND point_in_time <= '{end}'
    """.format(start=start, end=end)

    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()

    df = pd.DataFrame(results, columns=all_params+transformer_params)

    df = sqlio.read_sql_query(sql, conn)
    #df.loc[:, 'geom'] = df.loc[:, 'geom'].apply(wkt.loads)
    df.loc[:, 'geom'] = df.loc[:, 'geom'].apply(str)

    #df = gpd.GeoDataFrame(dxf, geometry='geom')

    return dd.from_pandas(df, npartitions=1)

def create_dataset_energiateollisuus(db_params, start, end, meta_params, geom_params, storm_params, outage_params, transformer_params, all_params):
    """ Gather dataset from db """
    #print('Reading data for {}-{}...'.format(start, end))

    conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (db_params['database'], db_params['user'], db_params['host'], db_params['password']))

    sql = """
        SELECT
        """
    first = True
    for p in meta_params:
        if not first:
            sql += ','
        first = False
        sql += "a.{}".format(p)

    for p in geom_params:
        sql += ',ST_AsText(a.{}) as {}'.format(p, p)

    for p in storm_params:
            sql += ',"{}"'.format(p)
    for p in outage_params:
        sql += ',c.{}'.format(p)

    sql += """
        FROM
        sasse.stormcell a
        INNER JOIN sasse.stormcell_features b ON a.id = b.polygon_id
        LEFT JOIN (
            SELECT
                b.id,
                SUM(transformers) AS outages,
                SUM(clients) AS customers
        FROM
            sasse.ene_outages aa,
            sasse.stormcell b,
            sasse.regions c
        WHERE
            date_trunc('hour', aa.start AT TIME ZONE 'Europe/Helsinki' AT TIME ZONE 'UTC') + interval '1 hour' = point_in_time
            AND ST_Intersects(st_setsrid (b.geom, 4326), st_setsrid (c.geom, 4326))
            AND aa.area = c.aluetunnus
            AND b.point_in_time > '{start}'
            AND b.point_in_time <= '{end}'
        GROUP BY
            b.id) c ON c.id = a.id
        WHERE
        st_intersects(ST_MakeEnvelope(20.6, 59.8, 31.5, 70.2, 4326), st_setsrid (a.geom, 4326))
        AND point_in_time > '{start}'
        AND point_in_time <= '{end}'
    """.format(start=start, end=end)

    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()

    df = pd.DataFrame(results, columns=all_params)

    df = sqlio.read_sql_query(sql, conn)
    #df.loc[:, 'geom'] = df.loc[:, 'geom'].apply(wkt.loads)
    df.loc[:, 'geom'] = df.loc[:, 'geom'].apply(str)

    #df = gpd.GeoDataFrame(df, geometry='geom')

    return df


def save_dataset(df, db_params, table_name='classification_dataset'):
    """
    Save classification dataset into the db

    df : DataFrame
         DataFrame data
    """
    if df is None or len(df) < 1:
        return

    print('Storing classification set to db sasse.{}...'.format(table_name))

    # host='docker.for.mac.localhost'
    host = db_params['host']

    # db_name, db_user, db_host, db_pass
    engine = create_engine('postgresql://{user}:{passwd}@{host}:5432/{db}'.format(user=db_params['user'],
                                                                                  passwd=db_params['password'],
                                                                                  host=host,
                                                                                  db=db_params['database']))

    df.to_sql(table_name, engine, schema='sasse', if_exists='append', index=False)


def _config(config_filename, config_name, param_section):
    # Read a yaml configuration file from disk
    with open(config_filename) as conf_file:
        config_dict = yaml.safe_load(conf_file)

    params = config_dict[param_section]

    return config_dict[config_name], params


def params_to_list(params, shortnames=False):
    """ Return list of queryable params """
    lst = []
    for param, info in params.items():
        for f in info['aggregation']:
            try:
                func, attr = f.split(',')
            except ValueError:
                func, attr = f, None

            if shortnames:
                if attr is not None:
                    lst.append(attr+' '+func[1:]+' '+info['name'])
                else:
                    lst.append(f[1:]+' '+info['name'])
            else:
                if attr is not None:
                    lst.append(func+'{'+attr+';'+param+'}')
                else:
                    lst.append(f+'{'+param+'}')

    return lst

def get_forest_data(config, params, wkt):
    """ Read forest data for given wkt """

    paramlist = params_to_list(params)
    # 0.05 degree --> ~2.75 km in longitude, ~5.5 km in latitude
    # 0.1 degree --> ~5.5 km in longitude, ~11 km in latitude
    url = "{host}/timeseries?format=json&starttime=data&endtime=data&param={params}&wkt={wkt}".format(host=config['host'],
                                                                                                      params=','.join(paramlist),
                                                                                                      wkt=wkt.simplify(0.1, preserve_topology=True))

    logging.debug(url)

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()[0]
    else:
        logging.error('Error with url: {}. Headers: {}'.format(url, response.headers))
        values = []
        for p in params:
           values.append(np.nan)
           return values

    met_params = {}
    values = []
    for param, value in data.items():
        f = re.search('(?<=@).*(?={)', param).group()
        p = re.search(r'(?<={).*(?=})', param).group()
        try:
            attr, func = p.split(';')
            name = attr+' '+func
        except ValueError:
            attr, func = None, p
            name = f

        met_params[name+' '+params[func]['name']] = float(value)
        values.append(float(value))

    # Throttle number of requests
    return pd.Series(met_params)
    #return values


def stats(geoms, config, params):
    """Get forest information"""

    #config, params = _config(config_filename, config_name, param_section)
    paramlist = params_to_list(params, True)
    rows = []
    for geom in geoms:
        rows.append(get_forest_data(config, params, wkt.loads(geom)))

    return pd.DataFrame(rows, columns=paramlist, index=geoms.index)

def process_time_range(start, end, dataset, db_params, met_params, meta_params, geom_params, storm_params, outage_params, transformers_params, all_params, config, forest_params, dataset_table):
    """ Process time range """

    paramlist = params_to_list(forest_params)

    print('Reading data for {}-{}'.format(start, end))

    dataset = []
    try:
        if dataset == 'loiste_jse':
            dataset = create_dataset_loiste_jse(db_params, start, end, meta_params, geom_params, storm_params, outage_params, transformers_params, all_params)
        else:
            dataset = create_dataset_energiateollisuus(db_params, start, end, meta_params, geom_params, storm_params, outage_params, transformers_params, all_params)

        dataset = dataset.compute()
    except psycopg2.OperationalError as e:
        print(e)
        if dataset == 'loiste_jse':
            dataset = create_dataset_loiste_jse(db_params, start, end, meta_params, geom_params, storm_params, outage_params, transformers_params, all_params)
        else:
            dataset = create_dataset_energiateollisuus(db_params, start, end, meta_params, geom_params, storm_params, outage_params, transformers_params, all_params)

        dataset = dataset.compute()

    #print(dataset)
    print('Reading data from DB done. Found {} records'.format(len(dataset)))

    if len(dataset) < 1 :
        return 0

    try:
        forest_data = dataset.geom.apply(lambda row: get_forest_data(config, forest_params, wkt.loads(row)))
    except AttributeError:
        return dataset

    #forest_data = pd.DataFrame(forest_data_rows, columns=paramlist, index=dataset.index)

    dataset = dataset.join(forest_data)

    print('\nDone. Found {} records'.format(dataset.shape[0]))

    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].fillna(0)
    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].astype(int)

    # Drop storm objects without customers or transformers, they are outside the range
    if options.dataset == 'loiste_jse':
        dataset.dropna(axis=0, subset=['all_customers', 'transformers'], inplace=True)

    # Drop rows with missing meteorological params
    for p in met_params:
        dataset = dataset[dataset[p] != -999]

    dataset.sort_values(by=['outages'], inplace=True)

    # Cast classes

    # outages
    limits = [(0,0), (1,2), (3,10), (11, 9999999)]
    i = 0
    for low, high in limits:
        dataset.loc[(dataset.loc[:, 'outages'] >= low) & (dataset.loc[:, 'outages'] <= high), 'class'] = i
        i += 1

    # customers
    limits = [(0,0), (1,250), (251,500), (501, 9999999)]
    i = 0
    for low, high in limits:
        dataset.loc[(dataset.loc[:, 'customers'] >= low) & (dataset.loc[:, 'customers'] <= high), 'class_customers'] = i
        i += 1

    dataset.loc[:, ['class', 'class_customers']] = dataset.loc[:, ['class', 'class_customers']].astype(int)

    dataset.drop(columns=['geom'], inplace=True)

    print("dataset:\n{}".format(dataset.head(1)))
    print("\n{}".format(list(dataset.dtypes)))
    print("\n{}".format(dataset.shape))

    # Save
    try:
        save_dataset(dataset, db_params, table_name=dataset_table)
    except BrokenPipeError as e:
        logging.warning(e)
        save_dataset(dataset, db_params, table_name=dataset_table)

    return len(dataset)

def main():

    #ssh = SmartMetHandler(options.smartmet_config_filename, options.smartmet_config_name, sleep_time=options.requests_throttle_time, param_section='forest_params')

    if hasattr(options, 'dask'):
        client = Client('{}:8786'.format(options.dask))
    else:
        client = Client()

    #print(get_forest_params('cnf/smartmet.yaml', False))
    #print(get_forest_params('cnf/smartmet.yaml', True))
    #print(client.run(get_version))
    #sys.exit()

    client.get_versions(check=True)
    logging.info(client)

    db_params = db_config(options.db_config_filename, options.db_config_name)

    s3 = boto3.resource('s3')

    # Load params
    content_object = s3.Object(conf_bucket, conf_file)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    config_dict = yaml.load(file_content, Loader=yaml.FullLoader)

    params = config_dict['params']
    met_params = set()
    shortnames = True
    for param, info in params.items():
        for f in info['aggregation']:
            if shortnames:
                met_params.add(f[1:]+' '+info['name'])
            else:
                met_params.add(f+'{'+param+'}')
    met_params = list(met_params)

    polygon_params = ['speed_self', 'angle_self', 'area_m2', 'area_diff']
    meta_params = ['id', 'storm_id', 'point_in_time', 'weather_parameter', 'low_limit', 'high_limit']
    geom_params = ['geom']
    outage_params = ['outages', 'customers']
    transformers_params = ['transformers', 'all_customers']

    storm_params = polygon_params + met_params

    all_params = meta_params + geom_params + storm_params + outage_params

    # Read data from database

    starttime = datetime.datetime.strptime(options.starttime, "%Y-%m-%d")
    endtime = datetime.datetime.strptime(options.endtime, "%Y-%m-%d")

    config, forest_params = _config(options.smartmet_config_filename, options.smartmet_config_name, 'forest_params')
    metas = {}
    for param in params_to_list(forest_params, True):
        metas[param] = 'float'

    dfs = []
    start = starttime
    while start <= endtime:
        end = start + timedelta(days=1)
        #start, end, dataset, db_params, meta_params, geom_params, storm_params, outage_params, transformers_params, all_paramm, config, forest_params, dataset_table
        dfs.append(client.submit(process_time_range, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), options.dataset, db_params, met_params, meta_params, geom_params, storm_params, outage_params, transformers_params, all_params, config, forest_params, options.dataset_table))

        start = end
        if end > endtime: end = endtime

    for i, d in enumerate(dfs):
        logging.info(client.gather(d))
        #logging.info('Processed {} records'.format(count))



if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default='cnf/rfc.ini', help='Config filename for training config')
    parser.add_argument('--config_name', type=str, default='test', help='Config section for training config')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--dataset', type=str, default='loiste_jse', help='Which dataaset to create (loiste_jse|energiateollisuus)')
    parser.add_argument('--dataset_table', type=str, default='classification_dataset_jse_forest', help='DB tablename')
    parser.add_argument('--starttime', type=str, default='2010-01-01', help='start time')
    parser.add_argument('--endtime', type=str, default='2019-01-01', help='end time')
    parser.add_argument('--smartmet_config_filename', type=str, default='cnf/smartmet.yaml', help='CNF file containing SmartMet Server pararemters')
    parser.add_argument('--smartmet_config_name', type=str, default='dev', help='Section name for smartmet')
    parser.add_argument('--requests_throttle_time', type=int, default=0, help='Sleep time after each SmartMet requests to throttle requests')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()
    read_options(options)

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    if hasattr(options, 'dask'):
        dask_info = options.dask
    else:
        dask_info = 'local'

    logging.info('Using config {} from {}'.format(options.config_name, options.config_filename))
    logging.info('DB config: {} | Dask: {} | Dataset: {} | Dataset db table: {} | Starttime: {} | Endtime: {}'.format(options.db_config_name, dask_info, options.dataset, options.dataset_table, options.starttime, options.endtime))
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)

    conf_bucket  ='fmi-sasse-cloudformation'
    conf_file    = 'smartmet.yaml'
    loiste_bbox  = '25,62.7,31.4,66.4'
    sssoy_bbox   = '24.5,60,30.6,63.5'

    main()
