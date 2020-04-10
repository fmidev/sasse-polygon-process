# -*- coding: utf-8 -*-
import sys, argparse, logging, joblib
sys.path.insert(0, 'lib/')
import datetime, time, boto3, yaml, os

from datetime import timedelta

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

from dask.distributed import Client, progress
import dask.dataframe as dd
import dask
from dask import delayed

from config import read_options

def save_dataset(df, table_name='classification_dataset'):
    """
    Save classification dataset into the db

    df : DataFrame
         DataFrame data
    """
    if df is None or len(df) < 1:
        return
    logging.info('Storing classification set to db sasse.{}...'.format(table_name))

    # db_name, db_user, db_host, db_pass
    engine = create_engine('postgresql://{user}:{passwd}@{host}:5432/{db}'.format(user=db_user,
                                                                                 passwd=db_pass,
                                                                                 host=db_host,
                                                                                 db=db_name))

    index_name = 'id'
    df.to_sql(table_name, engine, schema='sasse', if_exists='append', index=False)

def load_dataset(table_name='classification_dataset'):
    """
    Load classification dataset from db

    table_name : str
                 table name to be used
    """
    all_params_w_labels = all_params + ['class', 'class_customers']
    conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (db_name, db_user, db_host, db_pass))
    sql = """SELECT "{}" FROM sasse.{}""".format('","'.join(all_params_w_labels), table_name)

    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=all_params_w_labels)
    df.set_index('point_in_time', inplace=True)
    return df

# At the end, these were never used
def download_file(bucket, key, local_file):
    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    s3.download_file(bucket, key, local_file)

def get_keys(bucket, path, ending='tif', prefix=None):
    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    result = s3.list_objects(Bucket=bucket, Prefix=path)
    keys = []

    if prefix is None:
        prefix='s3://{}/'.format(bucket)

    for key in result['Contents']:
        if key['Key'][-len(ending):] == ending:
            keys.append('{}{}'.format(prefix, key['Key']))
    return keys

def get_dataset(start, end, meta_params, geom_params, storm_params, outage_params, transformer_params, all_params, paraller=True):
    """ Gather dataset from db """
    #print('Reading data for {}-{}...'.format(start, end))
    conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (db_name, db_user, db_host, db_pass))
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

    print('.', end='')

    df = pd.DataFrame(results, columns=all_params+transformer_params)

    return df



def main():

    client = Client('{}:8786'.format(options.dask))
    client.get_versions(check=True)

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
    starttime = datetime.datetime.strptime('2010-01-01', "%Y-%m-%d")
    endtime = datetime.datetime.strptime('2010-01-03', "%Y-%m-%d")

    logging.info('Reading data for {}-{}'.format(starttime, endtime))

    dfs, df = [], []
    start = starttime
    while start <= endtime:

        end = start + timedelta(days=1)

        dfs.append(delayed(get_dataset)(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, geom_params, storm_params, outage_params, transformers_params, all_params))

        start = end

    df = dask.compute(*dfs)
    progress(df)

    dataset = pd.concat(df)

    dataset.sort_values(by=['point_in_time'], inplace=True)
    logging.info('\nDone. Found {} records'.format(dataset.shape[0]))

    # Drop storm objects without customers or transformers, they are outside the range
    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].fillna(0)
    dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].astype(int)
    dataset.dropna(axis=0, subset=['all_customers', 'transformers'], inplace=True)

    # Drop rows with missing meteorological params
    for p in met_params:
        dataset = dataset[dataset[p] != -999]

    dataset.sort_values(by=['outages'], inplace=True)

    # Get forest information
    g = dataset['geom'].apply(wkt.loads)
    g_dataset = gpd.GeoDataFrame(dataset, geometry=g)

    paths = [('Forest FRA', 's3://fmi-asi-data-puusto/luke/2017/fra_luokka/puusto_fra_luokka_suomi_4326.tif'),
         ('Forest age', 's3://fmi-asi-data-puusto/luke/2017/ika/puusto_ika_suomi_4326.tif'),
         ('Forest site fertility', 's3://fmi-asi-data-puusto/luke/2017/kasvupaikka/puusto_kasvupaikka_suomi_4326.tif'),
         ('Forest stand mean diameter', 's3://fmi-asi-data-puusto/luke/2017/keskilapimitta/puusto_keskilapimitta_suomi_4326.tif'),
         ('Forest stand mean height', 's3://fmi-asi-data-puusto/luke/2017/keskipituus/puusto_keskipituus_suomi_4326.tif'),
         ('Forest canopy cover', 's3://fmi-asi-data-puusto/luke/2017/latvusto/puusto_latvusto_suomi_4326.tif'),
         ('Forest site main class', 's3://fmi-asi-data-puusto/luke/2017/paatyyppi/puusto_paatyyppi_suomi_4326.tif')
        ]
    paths = [('Forest FRA', 's3://fmi-asi-data-puusto/luke/2017/fra_luokka/puusto_fra_luokka_suomi_4326.tif')]
    chunks = {'y': 5000, 'x': 5000}

    ars = []
    for name, path in paths:
        ars.append((name, xr.open_rasterio(path, chunks=chunks)))

    def stats(row, data):
        bounds = wkt.loads(row.geom).bounds
        var_mean = data.sel(x=slice(bounds[0],bounds[2]), y=slice(bounds[3], bounds[1])).mean()
        var_max = data.sel(x=slice(bounds[0],bounds[2]), y=slice(bounds[3], bounds[1])).max()
        var_std = data.sel(x=slice(bounds[0],bounds[2]), y=slice(bounds[3], bounds[1])).std()
        return var_mean.values, var_max.values, var_std.values

    logging.info('Reading forest information...')
    for name, ar in ars:
        g_dataset['mean {}'.format(name)], g_dataset['max {}'.format(name)], g_dataset['std {}'.format(name)] = zip(*g_dataset.apply(lambda x: stats(x, ar), axis=1))
    logging.info('done')

    logging.info(g_dataset.columns.values)
    logging.info("\n.{}".format(g_dataset.head(2)))
    logging.info('GeoDataFrame shape: {}'.format(g_dataset.shape))

    # Convert back to original pandas DataFrame
    dataset = pd.DataFrame(g_dataset.drop(columns=['geom', 'geometry']))
    print('Dataset:')
    print(dataset.head(1))
    print(dataset.columns)
    print(dataset.dtypes)
    print(dataset.shape)

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
    logging.info("dataset:\n{}".format(dataset.head(1)))

    # Save
    save_dataset(dataset, table_name='classification_dataset_jse_forest')






if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default='cnf/rfc.ini', help='Config filename for training config')
    parser.add_argument('--config_name', type=str, default='test', help='Config section for training config')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()
    read_options(options)

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    logging.info('Using config {} from {}'.format(options.config_name, options.config_filename))

    db_host = '***REMOVED***'    
    db_port = 5432
    db_name = "postgres"
    db_user = "***REMOVED***"
    db_pass = "***REMOVED***"
    conf_bucket  ='fmi-sasse-cloudformation'
    conf_file    = 'smartmet.yaml'
    loiste_bbox  = '25,62.7,31.4,66.4'
    sssoy_bbox   = '24.5,60,30.6,63.5'

    main()
