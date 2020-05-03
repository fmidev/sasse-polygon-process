# -*- coding: utf-8 -*-
import sys, argparse, logging, joblib
sys.path.insert(0, 'lib/')
import datetime, time, boto3, yaml, os
from pathlib import Path

from datetime import timedelta

from configparser import ConfigParser

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import linalg

import geopandas as gpd
import rasterio
from shapely import wkt
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.rio.clip import clip
from rasterio import features
import xarray as xr
#import regionmask
from shapely.geometry import Polygon, MultiPolygon

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

    logging.info('Storing classification set to db sasse.{}...'.format(table_name))

    host = 'docker.for.mac.localhost'
    host = db_params['host']

    # db_name, db_user, db_host, db_pass
    engine = create_engine('postgresql://{user}:{passwd}@{host}:5432/{db}'.format(user=db_params['user'],
                                                                                  passwd=db_params['password'],
                                                                                  host=host,
                                                                                  db=db_params['database']))

    df.to_sql(table_name, engine, schema='sasse', if_exists='append', index=False)

def get_version():
    return rasterio.__version__

def stats(geoms, meta, data):
    """Get forest information"""

    # def dask_percentile(arr, axis=0, q=95):
    #     if len(arr.chunks[axis]) > 1:
    #         raise ValueError('Input array cannot be chunked along the percentile dimension.')
    #     return dask_array.map_blocks(np.percentile, arr, axis=axis, q=q, drop_axis=axis)
    #
    # def percentile(arr, axis=0, q=95):
    #     if isinstance(arr, dask_array.Array):
    #         return dask_percentile(arr, axis=axis, q=q)
    #     else:
    #         return np.percentile(arr, axis=axis, q=q)

    def x_slice(bounds, ar):
        return slice(max(bounds[0], min(ar.x).values), min(bounds[2], max(ar.x).values))
    def y_slice(bounds, ar):
        return slice(max(bounds[3], min(ar.y).values), min(bounds[1], max(ar.y).values))

    rows = []
    for geom in geoms:
        bounds = wkt.loads(geom).bounds
        data_bbox = data.sel(x=x_slice(bounds, data), y=y_slice(bounds, data))
        data_bbox = data_bbox.where(data_bbox < 32766)

        #var_median = data.quantile(.5, dim='band')
        #var_median = data.reduce(percentile, dim='band', q=50, allow_lazy=True)
        #var_9q = data.sel(x=slice(bounds[0],bounds[2]), y=slice(bounds[3], bounds[1])).quantile(.9)
        #var_1q = data.sel(x=slice(bounds[0],bounds[2]), y=slice(bounds[3], bounds[1])).quantile(.1)
        try:
            rows.append([float(data_bbox.mean().values), float(data_bbox.max().values), float(data_bbox.max().values)])
        except ValueError as e:
            logging.warning(e)
            rows.append([0, 0, 0])

    return pd.DataFrame(rows, columns=list(meta.keys()), index=geoms.index)

def get_file(path):
    """ Download file to /tmp if doesn't exist"""

    local_name = '/tmp/{}'.format(Path(path).name)
    local_file = Path(local_name)
    if not local_file.is_file():
        print('Downloading file {}'.format(path))
        s3_client = boto3.client('s3')
        s3_client.download_file('fmi-asi-data-puusto', path, local_name)

    return local_name

def main():

    if hasattr(options, 'dask'):
        client = Client('{}:8786'.format(options.dask))
    else:
        client = Client()

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

    logging.info('Reading data for {}-{}'.format(starttime, endtime))

    dfs, df = [], []
    start = starttime
    while start <= endtime:

        end = start + timedelta(days=1)

        if options.dataset == 'loiste_jse':
            #dfs.append(delayed(create_dataset_loiste_jse)(db_params, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, geom_params, storm_params, outage_params, transformers_params, all_params))
            dfs.append(client.submit(create_dataset_loiste_jse, db_params, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, geom_params, storm_params, outage_params, transformers_params, all_params))
        else:
            dfs.append(client.submit(create_dataset_energiateollisuus, db_params, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, geom_params, storm_params, outage_params, transformers_params, all_params))

        start = end

    for i, d in enumerate(dfs):
        try:
            dfs[i] = client.gather(d)
        except psycopg2.OperationalError as e:
            logging.error(e)
            dfs[i] = client.gather(d)

    with ProgressBar():
        df = dask.compute(*dfs)

    dataset = pd.concat(df)

    logging.info('Reading data from DB done. Found {} records'.format(len(dataset)))

    paths = [
        #('Forest FRA', 's3://fmi-asi-data-puusto/luke/2017/fra_luokka/puusto_fra_luokka_suomi_4326.tif'),
        ('Forest age', 's3://fmi-asi-data-puusto/luke/2017/ika/ika_suomi_4326_lowres.tif'),
        ('Forest site fertility', 's3://fmi-asi-data-puusto/luke/2017/kasvupaikka/kasvupaikka_suomi_4326_lowres.tif'),
        ('Forest stand mean diameter', 's3://fmi-asi-data-puusto/luke/2017/keskilapimitta/keskilapimitta_suomi_4326_lowres.tif'),
        ('Forest stand mean height', 's3://fmi-asi-data-puusto/luke/2017/keskipituus/keskipituus_suomi_4326_lowres.tif'),
        ('Forest canopy cover', 's3://fmi-asi-data-puusto/luke/2017/latvusto/latvusto_suomi_4326_lowres.tif'),
        ('Forest site main class', 's3://fmi-asi-data-puusto/luke/2017/paatyyppi/paatyyppi_suomi_4326_lowres.tif')
        ]

    #paths = [('Forest canopy cover', 's3://fmi-asi-data-puusto/luke/2017/latvusto/puusto_latvusto_suomi_4326.tif')]
    chunks = {'y': 5000, 'x': 5000}

    ars = []
    for name, path in paths:
        #filename = get_file(path)
        ars.append((name, xr.open_rasterio(path, chunks=chunks)))

    # Initiate forest data columns
    operations = ['mean', 'max', 'std']
    metas = {}
    for name, path in paths:
        meta = {}
        for op in operations:
            opname = '{} {}'.format(op, name)
            #dataset[opname] = np.nan
            meta[opname] = 'float'
        metas[name] = meta

    df = dd.from_pandas(dataset, npartitions=50)

    client.scatter(ars)
    client.scatter(df)

    with ProgressBar():
        #dataset = df.apply(lambda row: delayed(stats)(row, ars), axis=1)
        for name, ar in ars:
            forest_data = df.geom.map_partitions(stats, metas[name], ar, meta=pd.DataFrame(metas[name], index=df.index)).compute().reset_index(drop=True)
            dataset = dataset.reset_index(drop=True).join(forest_data)

    logging.info('\nDone. Found {} records'.format(dataset.shape[0]))

    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].replace('None', np.nan)
    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].fillna(0)
    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].astype(float)
    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].astype(int)
    #print(dataset.loc[:, ['outages', 'customers']])
    #print('--')

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

    #print(dataset.loc[:, ['class', 'class_customers']])
    #dataset.loc[:, ['class', 'class_customers']] = dataset.loc[:, ['class', 'class_customers']].fillna(0)
    dataset.fillna(0, inplace=True)
    dataset.loc[:, ['class', 'class_customers']] = dataset.loc[:, ['class', 'class_customers']].astype(float)
    dataset.loc[:, ['class', 'class_customers']] = dataset.loc[:, ['class', 'class_customers']].astype(int)

    dataset.drop(columns=['geom'], inplace=True)

    logging.info("dataset:\n{}".format(dataset.head(1)))
    logging.info("\n{}".format(dataset.dtypes))
    logging.info("\n{}".format(dataset.shape))

    # Save
    try:
        save_dataset(dataset, db_params, table_name=options.dataset_table)
    except BrokenPipeError as e:
        logging.warning(e)
        save_dataset(dataset, db_params, table_name=options.dataset_table)




if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default='cnf/rfc.ini', help='Config filename for training config')
    parser.add_argument('--config_name', type=str, default='test', help='Config section for training config')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--dataset', type=str, default='loiste_jse', help='Which dataaset to create (loiste_jse|energiateollisuus)')
    parser.add_argument('--dataset_table', type=str, default='classification_dataset_jse_forest', help='DB tablename')
    parser.add_argument('--starttime', type=str, default='2010-01-01', help='start time')
    parser.add_argument('--endtime', type=str, default='2019-01-01', help='end time')

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
