## Initialisation

Create and visualise dataset from DB. 

First some technical initialisations:


```python
!conda update -y dask
```

    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.12
      latest version: 4.8.1
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    ## Package Plan ##
    
      environment location: /home/ec2-user/anaconda3/envs/python3
    
      added / updated specs: 
        - dask
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        dask-2.10.0                |             py_0          12 KB
        dask-core-2.10.0           |             py_0         595 KB
        distributed-2.10.0         |             py_0         424 KB
        ------------------------------------------------------------
                                               Total:         1.0 MB
    
    The following packages will be UPDATED:
    
        dask:        2.9.2-py_0 --> 2.10.0-py_0
        dask-core:   2.9.2-py_0 --> 2.10.0-py_0
        distributed: 2.9.3-py_0 --> 2.10.0-py_0
    
    
    Downloading and Extracting Packages
    dask-2.10.0          | 12 KB     | ##################################### | 100% 
    dask-core-2.10.0     | 595 KB    | ##################################### | 100% 
    distributed-2.10.0   | 424 KB    | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done



```python
import pandas as pd
import psycopg2
from dask.distributed import Client
import dask.dataframe as dd
import dask
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
import numpy as np
from scipy.stats import pearsonr
import boto3, yaml
from sagemaker import get_execution_role
from dask import delayed
from datetime import timedelta
from sqlalchemy import create_engine
from ipywidgets import IntProgress
from IPython.display import display
import time
import matplotlib.ticker as mticker

sns.set()
sns.set_style("whitegrid")
%matplotlib inline
```


```python
client = Client('Dask-Scheduler.local-dask:8786')
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/distributed/client.py:1071: VersionMismatchWarning: Mismatched versions found
    
    blosc
    +--------------------------+---------+
    |                          | version |
    +--------------------------+---------+
    | client                   | None    |
    | scheduler                | 1.7.0   |
    | tcp://172.31.0.48:9000   | 1.7.0   |
    | tcp://172.31.1.97:9000   | 1.7.0   |
    | tcp://172.31.10.246:9000 | 1.7.0   |
    | tcp://172.31.13.126:9000 | 1.7.0   |
    | tcp://172.31.14.215:9000 | 1.7.0   |
    | tcp://172.31.15.161:9000 | 1.7.0   |
    | tcp://172.31.15.202:9000 | 1.7.0   |
    | tcp://172.31.24.58:9000  | 1.7.0   |
    | tcp://172.31.25.15:9000  | 1.7.0   |
    | tcp://172.31.26.201:9000 | 1.7.0   |
    | tcp://172.31.28.200:9000 | 1.7.0   |
    | tcp://172.31.28.76:9000  | 1.7.0   |
    | tcp://172.31.28.78:9000  | 1.7.0   |
    | tcp://172.31.35.192:9000 | 1.7.0   |
    | tcp://172.31.37.88:9000  | 1.7.0   |
    | tcp://172.31.42.122:9000 | 1.7.0   |
    | tcp://172.31.42.88:9000  | 1.7.0   |
    | tcp://172.31.43.211:9000 | 1.7.0   |
    | tcp://172.31.44.219:9000 | 1.7.0   |
    | tcp://172.31.44.247:9000 | 1.7.0   |
    | tcp://172.31.45.6:9000   | 1.7.0   |
    | tcp://172.31.46.187:9000 | 1.7.0   |
    | tcp://172.31.5.30:9000   | 1.7.0   |
    | tcp://172.31.6.95:9000   | 1.7.0   |
    | tcp://172.31.7.231:9000  | 1.7.0   |
    +--------------------------+---------+
    
    cloudpickle
    +--------------------------+---------+
    |                          | version |
    +--------------------------+---------+
    | client                   | 0.5.3   |
    | scheduler                | 1.2.2   |
    | tcp://172.31.0.48:9000   | 1.2.2   |
    | tcp://172.31.1.97:9000   | 1.2.2   |
    | tcp://172.31.10.246:9000 | 1.2.2   |
    | tcp://172.31.13.126:9000 | 1.2.2   |
    | tcp://172.31.14.215:9000 | 1.2.2   |
    | tcp://172.31.15.161:9000 | 1.2.2   |
    | tcp://172.31.15.202:9000 | 1.2.2   |
    | tcp://172.31.24.58:9000  | 1.2.2   |
    | tcp://172.31.25.15:9000  | 1.2.2   |
    | tcp://172.31.26.201:9000 | 1.2.2   |
    | tcp://172.31.28.200:9000 | 1.2.2   |
    | tcp://172.31.28.76:9000  | 1.2.2   |
    | tcp://172.31.28.78:9000  | 1.2.2   |
    | tcp://172.31.35.192:9000 | 1.2.2   |
    | tcp://172.31.37.88:9000  | 1.2.2   |
    | tcp://172.31.42.122:9000 | 1.2.2   |
    | tcp://172.31.42.88:9000  | 1.2.2   |
    | tcp://172.31.43.211:9000 | 1.2.2   |
    | tcp://172.31.44.219:9000 | 1.2.2   |
    | tcp://172.31.44.247:9000 | 1.2.2   |
    | tcp://172.31.45.6:9000   | 1.2.2   |
    | tcp://172.31.46.187:9000 | 1.2.2   |
    | tcp://172.31.5.30:9000   | 1.2.2   |
    | tcp://172.31.6.95:9000   | 1.2.2   |
    | tcp://172.31.7.231:9000  | 1.2.2   |
    +--------------------------+---------+
    
    msgpack
    +--------------------------+---------+
    |                          | version |
    +--------------------------+---------+
    | client                   | 0.6.0   |
    | scheduler                | 0.6.1   |
    | tcp://172.31.0.48:9000   | 0.6.1   |
    | tcp://172.31.1.97:9000   | 0.6.1   |
    | tcp://172.31.10.246:9000 | 0.6.1   |
    | tcp://172.31.13.126:9000 | 0.6.1   |
    | tcp://172.31.14.215:9000 | 0.6.1   |
    | tcp://172.31.15.161:9000 | 0.6.1   |
    | tcp://172.31.15.202:9000 | 0.6.1   |
    | tcp://172.31.24.58:9000  | 0.6.1   |
    | tcp://172.31.25.15:9000  | 0.6.1   |
    | tcp://172.31.26.201:9000 | 0.6.1   |
    | tcp://172.31.28.200:9000 | 0.6.1   |
    | tcp://172.31.28.76:9000  | 0.6.1   |
    | tcp://172.31.28.78:9000  | 0.6.1   |
    | tcp://172.31.35.192:9000 | 0.6.1   |
    | tcp://172.31.37.88:9000  | 0.6.1   |
    | tcp://172.31.42.122:9000 | 0.6.1   |
    | tcp://172.31.42.88:9000  | 0.6.1   |
    | tcp://172.31.43.211:9000 | 0.6.1   |
    | tcp://172.31.44.219:9000 | 0.6.1   |
    | tcp://172.31.44.247:9000 | 0.6.1   |
    | tcp://172.31.45.6:9000   | 0.6.1   |
    | tcp://172.31.46.187:9000 | 0.6.1   |
    | tcp://172.31.5.30:9000   | 0.6.1   |
    | tcp://172.31.6.95:9000   | 0.6.1   |
    | tcp://172.31.7.231:9000  | 0.6.1   |
    +--------------------------+---------+
    
    numpy
    +--------------------------+---------+
    |                          | version |
    +--------------------------+---------+
    | client                   | 1.14.3  |
    | scheduler                | 1.18.1  |
    | tcp://172.31.0.48:9000   | 1.18.1  |
    | tcp://172.31.1.97:9000   | 1.18.1  |
    | tcp://172.31.10.246:9000 | 1.18.1  |
    | tcp://172.31.13.126:9000 | 1.18.1  |
    | tcp://172.31.14.215:9000 | 1.18.1  |
    | tcp://172.31.15.161:9000 | 1.18.1  |
    | tcp://172.31.15.202:9000 | 1.18.1  |
    | tcp://172.31.24.58:9000  | 1.18.1  |
    | tcp://172.31.25.15:9000  | 1.18.1  |
    | tcp://172.31.26.201:9000 | 1.18.1  |
    | tcp://172.31.28.200:9000 | 1.18.1  |
    | tcp://172.31.28.76:9000  | 1.18.1  |
    | tcp://172.31.28.78:9000  | 1.18.1  |
    | tcp://172.31.35.192:9000 | 1.18.1  |
    | tcp://172.31.37.88:9000  | 1.18.1  |
    | tcp://172.31.42.122:9000 | 1.18.1  |
    | tcp://172.31.42.88:9000  | 1.18.1  |
    | tcp://172.31.43.211:9000 | 1.18.1  |
    | tcp://172.31.44.219:9000 | 1.18.1  |
    | tcp://172.31.44.247:9000 | 1.18.1  |
    | tcp://172.31.45.6:9000   | 1.18.1  |
    | tcp://172.31.46.187:9000 | 1.18.1  |
    | tcp://172.31.5.30:9000   | 1.18.1  |
    | tcp://172.31.6.95:9000   | 1.18.1  |
    | tcp://172.31.7.231:9000  | 1.18.1  |
    +--------------------------+---------+
    
    toolz
    +--------------------------+---------+
    |                          | version |
    +--------------------------+---------+
    | client                   | 0.9.0   |
    | scheduler                | 0.10.0  |
    | tcp://172.31.0.48:9000   | 0.10.0  |
    | tcp://172.31.1.97:9000   | 0.10.0  |
    | tcp://172.31.10.246:9000 | 0.10.0  |
    | tcp://172.31.13.126:9000 | 0.10.0  |
    | tcp://172.31.14.215:9000 | 0.10.0  |
    | tcp://172.31.15.161:9000 | 0.10.0  |
    | tcp://172.31.15.202:9000 | 0.10.0  |
    | tcp://172.31.24.58:9000  | 0.10.0  |
    | tcp://172.31.25.15:9000  | 0.10.0  |
    | tcp://172.31.26.201:9000 | 0.10.0  |
    | tcp://172.31.28.200:9000 | 0.10.0  |
    | tcp://172.31.28.76:9000  | 0.10.0  |
    | tcp://172.31.28.78:9000  | 0.10.0  |
    | tcp://172.31.35.192:9000 | 0.10.0  |
    | tcp://172.31.37.88:9000  | 0.10.0  |
    | tcp://172.31.42.122:9000 | 0.10.0  |
    | tcp://172.31.42.88:9000  | 0.10.0  |
    | tcp://172.31.43.211:9000 | 0.10.0  |
    | tcp://172.31.44.219:9000 | 0.10.0  |
    | tcp://172.31.44.247:9000 | 0.10.0  |
    | tcp://172.31.45.6:9000   | 0.10.0  |
    | tcp://172.31.46.187:9000 | 0.10.0  |
    | tcp://172.31.5.30:9000   | 0.10.0  |
    | tcp://172.31.6.95:9000   | 0.10.0  |
    | tcp://172.31.7.231:9000  | 0.10.0  |
    +--------------------------+---------+
    
    tornado
    +--------------------------+---------+
    |                          | version |
    +--------------------------+---------+
    | client                   | 5.0.2   |
    | scheduler                | 6.0.3   |
    | tcp://172.31.0.48:9000   | 6.0.3   |
    | tcp://172.31.1.97:9000   | 6.0.3   |
    | tcp://172.31.10.246:9000 | 6.0.3   |
    | tcp://172.31.13.126:9000 | 6.0.3   |
    | tcp://172.31.14.215:9000 | 6.0.3   |
    | tcp://172.31.15.161:9000 | 6.0.3   |
    | tcp://172.31.15.202:9000 | 6.0.3   |
    | tcp://172.31.24.58:9000  | 6.0.3   |
    | tcp://172.31.25.15:9000  | 6.0.3   |
    | tcp://172.31.26.201:9000 | 6.0.3   |
    | tcp://172.31.28.200:9000 | 6.0.3   |
    | tcp://172.31.28.76:9000  | 6.0.3   |
    | tcp://172.31.28.78:9000  | 6.0.3   |
    | tcp://172.31.35.192:9000 | 6.0.3   |
    | tcp://172.31.37.88:9000  | 6.0.3   |
    | tcp://172.31.42.122:9000 | 6.0.3   |
    | tcp://172.31.42.88:9000  | 6.0.3   |
    | tcp://172.31.43.211:9000 | 6.0.3   |
    | tcp://172.31.44.219:9000 | 6.0.3   |
    | tcp://172.31.44.247:9000 | 6.0.3   |
    | tcp://172.31.45.6:9000   | 6.0.3   |
    | tcp://172.31.46.187:9000 | 6.0.3   |
    | tcp://172.31.5.30:9000   | 6.0.3   |
    | tcp://172.31.6.95:9000   | 6.0.3   |
    | tcp://172.31.7.231:9000  | 6.0.3   |
    +--------------------------+---------+
      warnings.warn(version_module.VersionMismatchWarning(msg[0]["warning"]))



```python
client.get_versions(check=False)
client
```




<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://Dask-Scheduler.local-dask:8786</li>
  <li><b>Dashboard: </b><a href='http://Dask-Scheduler.local-dask:8787/status' target='_blank'>http://Dask-Scheduler.local-dask:8787/status</a>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Cluster</h3>
<ul style="text-align: left; list-style:none; margin: 0; padding: 0;">
  <li><b>Workers: </b>25</li>
  <li><b>Cores: </b>50</li>
  <li><b>Memory: </b>45.00 GB</li>
</ul>
</td>
</tr>
</table>



For loading data:


```python
db_host = '***REMOVED***'
db_port = 5432
db_name = "postgres"
db_user = "***REMOVED***"
db_pass = "***REMOVED***"
conf_bucket  ='fmi-sasse-cloudformation'
conf_file    = 'smartmet.yaml'
loiste_bbox  = '25,62.7,31.4,66.4'
sssoy_bbox   = '24.5,60,30.6,63.5'
```

Load params:


```python
s3 = boto3.resource('s3')

content_object = s3.Object(conf_bucket, conf_file)
file_content = content_object.get()['Body'].read().decode('utf-8')
config_dict = yaml.load(file_content)

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
```

Update polygon and meteorlogical params. Lists are used in process_polygons function and while saving the data to db.


```python
polygon_params = ['speed_self', 'angle_self', 'area_m2', 'area_diff']
meta_params = ['id', 'storm_id', 'point_in_time', 'weather_parameter', 'low_limit', 'high_limit']
outage_params = ['outages', 'customers']
transformers_params = ['transformers', 'all_customers']
storm_params = polygon_params + met_params
all_params = meta_params + storm_params + outage_params
```

Functions to save dataset (used later)


```python
def save_dataset(df, table_name='classification_dataset'):
    """
    Save classification dataset into the db

    df : DataFrame
         DataFrame data
    """
    if df is None or len(df) < 1:
        return
    print('Storing classification set to db sasse.{}...'.format(table_name))

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
```

## Creating and loading dataset

### Loiste + JSE

#### Create dataset

Get dataset from db. This is a long process. If the dataset is already saved, this may be skipped.


```python
def get_dataset(start, end, meta_params, storm_params, outage_params, transformer_params, all_params, paraller=True):
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

    for p in storm_params:
            sql += ',"{}"'.format(p)
    for p in outage_params:
        sql += ',c.{}'.format(p)
#    for p in transformer_params:
#        sql += ',d.{}'.format(p)

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
    
    df = pd.DataFrame(results, columns=all_params)
    
    return df    
```


```python
%%time
from dask.distributed import progress
starttime = datetime.datetime.strptime('2010-01-01', "%Y-%m-%d")
endtime = datetime.datetime.strptime('2019-01-01', "%Y-%m-%d")
paraller = True

print('Reading data for {}-{}'.format(starttime, endtime))
# Progress bar
max_count = 108 # would be better to calculate
f = IntProgress(min=0, max=max_count) # instantiate the bar
display(f) # display the bar

dfs, df = [], []
start = starttime
while start <= endtime:
    end = start + timedelta(days=1)
    if paraller: dfs.append(delayed(get_dataset)(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, storm_params, outage_params, transformers_params, all_params))
    else: df.append(get_dataset(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, storm_params, outage_params, transformers_params, all_params, paraller))
    start = end

if paraller:
    df = dask.compute(*dfs)
    progress(df)

dataset = pd.concat(df)
    
dataset.sort_values(by=['point_in_time'], inplace=True)
print('\nDone. Found {} records'.format(dataset.shape[0]))
```

    Reading data for 2010-01-01 00:00:00-2019-01-01 00:00:00



    IntProgress(value=0, max=108)


    
    Done. Found 28188 records
    CPU times: user 21.4 s, sys: 996 ms, total: 22.4 s
    Wall time: 2h 15min 46s


If outages and customers are none, they are 0. If all_customers or all_transformers is 0, the cell do not overlap the power grid. Those lines may be dropped.


```python
dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].fillna(0)
dataset.loc[:,['outages','customers']] = dataset.loc[:,['outages','customers']].astype(int)
dataset.dropna(axis=0, subset=['all_customers', 'transformers'], inplace=True)

# Drop rows with missing meteorological params
for p in met_params:
    dataset = dataset[dataset[p] != -999]

dataset.sort_values(by=['outages'], inplace=True)
```

#### Cast classes

Following plots help us to divide data to classes.


```python
fig, ax = plt.subplots(figsize=(20,5))
dataset.loc[(dataset['outages'] >= 1), :].groupby(by=['outages']).count()['id'].plot(kind='bar', title='Number of polygons by outages')
plt.ylabel('Number of polygons')
plt.grid(False)

fig, ax = plt.subplots(figsize=(20,5))
dataset.loc[(dataset['customers'] >= 1), :].groupby(by=['customers']).count()['id'].plot(kind='bar', title='Number of polygons by customers')
xtick_count = 15
every_nth = len(ax.xaxis.get_ticklabels()) // xtick_count

for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
plt.ylabel('Number of polygons')
plt.grid(False)

fig, ax = plt.subplots(figsize=(5,5))
dataset.plot(kind='scatter', x='outages', y='customers', ax=ax, c='green')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa7dd3bb38>




![png](output_25_1.png)



![png](output_25_2.png)



![png](output_25_3.png)


From above plots we can see that most of the polygons cause only 1 outage. It is notable that because reason of outages are not saved reliably, this bin may include a lot of noise. Outages are always reported from transformers. Second and third plot help us to give sense how many customers (namely houses) each outage means. 

First, if number of outages is 0, the class is 0. Rest of the dataset could be divided to classes following: 

Class 1:  1 - 2  outages  --> max ~50 customers without electricity<br>
Class 2:  3 - 10 outages  --> max ~250 customers without electricity <br>
Class 3: 11 -    outages  --> lots of customers without electricity<br>

We can also classify the data based on number of customers respectively:. 

Class 1:   1 - 250 cusomters  --> max ~2 outages<br>
Class 2: 251 - 500 customers  --> max ~10 outages <br>
Class 3: 501 -     customers  --> lots of outages<br>


```python
# outages
limits = [(0,0), (1,2), (3,10), (11, 9999999)]
i = 0
for low, high in limits:
    dataset.loc[(dataset.loc[:, 'outages'] >= low) & (dataset.loc[:, 'outages'] <= high), 'class'] = i
    i += 1
    
# outages
limits = [(0,0), (1,250), (251,500), (501, 9999999)]
i = 0
for low, high in limits:
    dataset.loc[(dataset.loc[:, 'customers'] >= low) & (dataset.loc[:, 'customers'] <= high), 'class_customers'] = i
    i += 1

dataset.loc[:, ['class', 'class_customers']] = dataset.loc[:, ['class', 'class_customers']].astype(int)
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>storm_id</th>
      <th>weather_parameter</th>
      <th>low_limit</th>
      <th>high_limit</th>
      <th>speed_self</th>
      <th>angle_self</th>
      <th>area_m2</th>
      <th>area_diff</th>
      <th>MIN Temperature</th>
      <th>...</th>
      <th>SUM Snowfall</th>
      <th>AVG Snowdepth</th>
      <th>AVG Mixed layer height</th>
      <th>MAX Precipitation kg/m2</th>
      <th>outages</th>
      <th>customers</th>
      <th>transformers</th>
      <th>all_customers</th>
      <th>class</th>
      <th>class_customers</th>
    </tr>
    <tr>
      <th>point_in_time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-11-22 16:00:00</th>
      <td>619845</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>45.529404</td>
      <td>22.158261</td>
      <td>880685907764</td>
      <td>-63472071911</td>
      <td>263.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>513.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3063</td>
      <td>40911</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2016-01-27 23:00:00</th>
      <td>666775</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>1.986536</td>
      <td>131.888932</td>
      <td>3259752117168</td>
      <td>-33907899104</td>
      <td>262.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>544.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-03-12 21:00:00</th>
      <td>883695</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>13.150008</td>
      <td>349.718322</td>
      <td>1806219642951</td>
      <td>-783799885</td>
      <td>261.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>315.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>199.0</td>
      <td>2173.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-10-24 06:00:00</th>
      <td>921162</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>13.193252</td>
      <td>2.062289</td>
      <td>3858873495496</td>
      <td>-33958081062</td>
      <td>262.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>535.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-01-02 00:00:00</th>
      <td>776131</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>15.020581</td>
      <td>310.955154</td>
      <td>1345851044258</td>
      <td>8924422476</td>
      <td>253.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>460.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1331</td>
      <td>21838</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-03-19 00:00:00</th>
      <td>777731</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>0.452255</td>
      <td>194.328946</td>
      <td>3105478695001</td>
      <td>-38627220511</td>
      <td>257.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>617.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-11-13 15:00:00</th>
      <td>398137</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>33.197548</td>
      <td>358.345354</td>
      <td>1140246727380</td>
      <td>-33539134878</td>
      <td>259.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>481.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2111.0</td>
      <td>14455.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2012-07-17 04:00:00</th>
      <td>302193</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>9.389725</td>
      <td>35.737227</td>
      <td>1044025512360</td>
      <td>-30495108834</td>
      <td>281.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>267.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>437</td>
      <td>7936</td>
      <td>98319</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2013-10-24 07:00:00</th>
      <td>399847</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>12.976426</td>
      <td>15.483334</td>
      <td>2808040248290</td>
      <td>-43657913308</td>
      <td>267.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>546.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>648</td>
      <td>13241</td>
      <td>160548</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-09-08 08:00:00</th>
      <td>301180</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>25.135854</td>
      <td>350.047659</td>
      <td>2200160608030</td>
      <td>-29090550284</td>
      <td>274.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>688.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>1429</td>
      <td>13241</td>
      <td>160548</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-07-31 11:00:00</th>
      <td>619523</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>34.962486</td>
      <td>16.475244</td>
      <td>543420852575</td>
      <td>-64116211901</td>
      <td>282.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>765.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>2052</td>
      <td>5111.0</td>
      <td>57971.0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-18 16:00:00</th>
      <td>302475</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>22.110171</td>
      <td>170.597424</td>
      <td>1081320700323</td>
      <td>64107961518</td>
      <td>280.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>604.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1117</td>
      <td>13241</td>
      <td>160548</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2010-01-03 18:00:00</th>
      <td>32276</td>
      <td>5607-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>45.885378</td>
      <td>316.657149</td>
      <td>241096441088</td>
      <td>15939045602</td>
      <td>252.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>497.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 22:00:00</th>
      <td>611183</td>
      <td>611054-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>6.078201</td>
      <td>0.972651</td>
      <td>31675028</td>
      <td>-230371049</td>
      <td>281.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>859.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>16.0</td>
      <td>100.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-25 21:00:00</th>
      <td>620073</td>
      <td>608707-WindGust-20-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>35.469293</td>
      <td>40.191461</td>
      <td>1735472585973</td>
      <td>-92904035307</td>
      <td>261.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>634.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>4154.0</td>
      <td>50728.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-21 22:00:00</th>
      <td>619777</td>
      <td>603716-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>15.703814</td>
      <td>321.853379</td>
      <td>3052647924801</td>
      <td>-69593631969</td>
      <td>262.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>469.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 12:00:00</th>
      <td>620173</td>
      <td>609586-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>6.033514</td>
      <td>237.253781</td>
      <td>2934198077095</td>
      <td>50604445512</td>
      <td>267.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>798.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-26 10:00:00</th>
      <td>620100</td>
      <td>609236-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>16.380546</td>
      <td>194.133716</td>
      <td>1577212789861</td>
      <td>24991088955</td>
      <td>263.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>835.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1568.0</td>
      <td>11325.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-26 06:00:00</th>
      <td>620092</td>
      <td>609052-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>6.589302</td>
      <td>190.097460</td>
      <td>1539445013269</td>
      <td>11172733009</td>
      <td>258.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>764.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2796.0</td>
      <td>20385.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 13:00:00</th>
      <td>620176</td>
      <td>610702-WindGust-20-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>4.012660</td>
      <td>231.204280</td>
      <td>2971332048904</td>
      <td>37133971809</td>
      <td>267.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>785.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-22 04:00:00</th>
      <td>619794</td>
      <td>605021-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>-999.000000</td>
      <td>32.887687</td>
      <td>1664194863154</td>
      <td>1664194863154</td>
      <td>261.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>563.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 10:00:00</th>
      <td>610566</td>
      <td>608994-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>55.638931</td>
      <td>212.502566</td>
      <td>881568368785</td>
      <td>1455122876</td>
      <td>269.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1012.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>617.0</td>
      <td>2437.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 05:00:00</th>
      <td>610309</td>
      <td>609236-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>39.064357</td>
      <td>1.260019</td>
      <td>1391983639635</td>
      <td>-90618905934</td>
      <td>269.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1142.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>12380.0</td>
      <td>149702.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-21 10:00:00</th>
      <td>619727</td>
      <td>604096-WindGust-20-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>18.227279</td>
      <td>21.666144</td>
      <td>3725023507539</td>
      <td>2056714445</td>
      <td>254.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>532.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-21 12:00:00</th>
      <td>619738</td>
      <td>604096-WindGust-20-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>5.991400</td>
      <td>4.657696</td>
      <td>3653198575532</td>
      <td>-57971628122</td>
      <td>254.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>548.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 19:00:00</th>
      <td>620194</td>
      <td>611054-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>11.912931</td>
      <td>201.026138</td>
      <td>3146391230156</td>
      <td>23132482749</td>
      <td>262.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>749.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-25 14:00:00</th>
      <td>620058</td>
      <td>608598-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>18.017911</td>
      <td>22.684965</td>
      <td>2200277604192</td>
      <td>-39435675575</td>
      <td>263.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>593.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3462.0</td>
      <td>43393.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 19:00:00</th>
      <td>611040</td>
      <td>609586-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>-999.000000</td>
      <td>219.583880</td>
      <td>500418837814</td>
      <td>500418837814</td>
      <td>277.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>853.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>109.0</td>
      <td>1182.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-22 06:00:00</th>
      <td>619802</td>
      <td>605021-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>17.031395</td>
      <td>28.974775</td>
      <td>1462153608130</td>
      <td>-98227162927</td>
      <td>260.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>567.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241</td>
      <td>160548</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2015-11-27 18:00:00</th>
      <td>620191</td>
      <td>611001-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>6.982892</td>
      <td>214.968662</td>
      <td>3123258747406</td>
      <td>25761124152</td>
      <td>263.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>747.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2012-09-18 00:00:00</th>
      <td>302836</td>
      <td>284109-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>2.009907</td>
      <td>358.033318</td>
      <td>1541965893868</td>
      <td>18658746275</td>
      <td>271.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>536.0</td>
      <td>0.0</td>
      <td>16</td>
      <td>11769</td>
      <td>5105.0</td>
      <td>57953.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-12-13 02:00:00</th>
      <td>378777</td>
      <td>377478-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>193.262336</td>
      <td>200.627121</td>
      <td>905787139333</td>
      <td>905696111976</td>
      <td>270.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1260.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>5696</td>
      <td>10871.0</td>
      <td>125298.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-12-13 02:00:00</th>
      <td>400281</td>
      <td>378695-WindGust-20-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>46.697155</td>
      <td>343.341947</td>
      <td>2077809612151</td>
      <td>-21129413084</td>
      <td>261.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>883.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>5696</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-17 09:00:00</th>
      <td>302231</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>13.592407</td>
      <td>0.987562</td>
      <td>909592064125</td>
      <td>-35229338158</td>
      <td>281.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>598.0</td>
      <td>0.0</td>
      <td>18</td>
      <td>4129</td>
      <td>5330</td>
      <td>59740</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-06-03 11:00:00</th>
      <td>556228</td>
      <td>508233-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>45.500675</td>
      <td>7.167652</td>
      <td>2176803769898</td>
      <td>-36247382612</td>
      <td>273.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>954.0</td>
      <td>0.0</td>
      <td>19</td>
      <td>1153</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-06-03 11:00:00</th>
      <td>508218</td>
      <td>507498-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>27.357702</td>
      <td>171.846102</td>
      <td>1362465206394</td>
      <td>38916104858</td>
      <td>272.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1659.0</td>
      <td>0.0</td>
      <td>19</td>
      <td>1153</td>
      <td>6380.0</td>
      <td>81347.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-10-13 08:00:00</th>
      <td>357836</td>
      <td>356365-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>132.950896</td>
      <td>327.491908</td>
      <td>1065948717350</td>
      <td>152049565845</td>
      <td>271.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1308.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>6307</td>
      <td>5051.0</td>
      <td>57732.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-06-03 12:00:00</th>
      <td>556285</td>
      <td>508349-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>43.870883</td>
      <td>7.469488</td>
      <td>2123398738102</td>
      <td>-53405031796</td>
      <td>273.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>969.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>1349</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-06-03 12:00:00</th>
      <td>508335</td>
      <td>507587-WindGust-20-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>118.002415</td>
      <td>188.721463</td>
      <td>1269381243125</td>
      <td>-93083963269</td>
      <td>273.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1609.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>1349</td>
      <td>7817.0</td>
      <td>100072.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2010-05-24 14:00:00</th>
      <td>33808</td>
      <td>27383-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>77.174407</td>
      <td>188.137052</td>
      <td>1176942964137</td>
      <td>165701207019</td>
      <td>278.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>963.0</td>
      <td>0.0</td>
      <td>26</td>
      <td>7173</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-10-13 05:00:00</th>
      <td>357510</td>
      <td>357341-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>84.686120</td>
      <td>329.053719</td>
      <td>778382870738</td>
      <td>52514131432</td>
      <td>270.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1421.0</td>
      <td>0.0</td>
      <td>27</td>
      <td>10617</td>
      <td>4669.0</td>
      <td>54840.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-18 15:00:00</th>
      <td>302468</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>29.314348</td>
      <td>161.518479</td>
      <td>1017212738805</td>
      <td>40204990080</td>
      <td>283.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>687.0</td>
      <td>0.0</td>
      <td>28</td>
      <td>7350</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2011-06-01 13:00:00</th>
      <td>91376</td>
      <td>90882-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>80578116087</td>
      <td>80578116087</td>
      <td>290.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1718.0</td>
      <td>0.0</td>
      <td>32</td>
      <td>9279</td>
      <td>2995.0</td>
      <td>30442.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-05-23 13:00:00</th>
      <td>565792</td>
      <td>551557-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>32.982817</td>
      <td>4.419663</td>
      <td>750528198082</td>
      <td>-8373451434</td>
      <td>272.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>955.0</td>
      <td>0.0</td>
      <td>33</td>
      <td>15884</td>
      <td>4499.0</td>
      <td>52255.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-05-23 13:00:00</th>
      <td>553675</td>
      <td>551557-WindGust-15-999</td>
      <td>WindGust</td>
      <td>20</td>
      <td>999</td>
      <td>79.607071</td>
      <td>14.261287</td>
      <td>48858471181</td>
      <td>-12167942666</td>
      <td>279.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1736.0</td>
      <td>0.0</td>
      <td>33</td>
      <td>15884</td>
      <td>4131.0</td>
      <td>50937.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-05-23 13:00:00</th>
      <td>553660</td>
      <td>551557-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>24.934307</td>
      <td>118.002760</td>
      <td>269309824337</td>
      <td>-26893453998</td>
      <td>276.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1599.0</td>
      <td>0.0</td>
      <td>34</td>
      <td>16961</td>
      <td>6598.0</td>
      <td>75413.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-10-13 06:00:00</th>
      <td>357603</td>
      <td>356571-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>78.535896</td>
      <td>326.479170</td>
      <td>812569622339</td>
      <td>34186751601</td>
      <td>270.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1404.0</td>
      <td>0.0</td>
      <td>37</td>
      <td>16646</td>
      <td>5111.0</td>
      <td>57971.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-18 13:00:00</th>
      <td>302455</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>2.928200</td>
      <td>19.118443</td>
      <td>924355457070</td>
      <td>15253045950</td>
      <td>282.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>822.0</td>
      <td>0.0</td>
      <td>39</td>
      <td>11242</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2010-05-24 17:00:00</th>
      <td>33819</td>
      <td>27383-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>47.427944</td>
      <td>176.098751</td>
      <td>1365868053936</td>
      <td>29137274394</td>
      <td>277.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>667.0</td>
      <td>0.0</td>
      <td>40</td>
      <td>10886</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-12-13 03:00:00</th>
      <td>378888</td>
      <td>377478-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>66.758718</td>
      <td>357.909716</td>
      <td>884833333318</td>
      <td>-20953806016</td>
      <td>269.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1293.0</td>
      <td>0.0</td>
      <td>41</td>
      <td>10167</td>
      <td>11609.0</td>
      <td>134983.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-12-13 03:00:00</th>
      <td>400285</td>
      <td>378695-WindGust-20-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>36.999693</td>
      <td>343.556301</td>
      <td>2056731884438</td>
      <td>-21077727712</td>
      <td>264.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>904.0</td>
      <td>0.0</td>
      <td>42</td>
      <td>10441</td>
      <td>13241.0</td>
      <td>160548.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2010-05-24 16:00:00</th>
      <td>33815</td>
      <td>28099-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>21.708197</td>
      <td>197.269295</td>
      <td>1336730779543</td>
      <td>58641269050</td>
      <td>277.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>771.0</td>
      <td>0.0</td>
      <td>42</td>
      <td>8632</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2011-06-01 11:00:00</th>
      <td>91220</td>
      <td>90882-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>109593620209</td>
      <td>109593620209</td>
      <td>291.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1843.0</td>
      <td>0.0</td>
      <td>42</td>
      <td>14832</td>
      <td>4357.0</td>
      <td>52150.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-18 14:00:00</th>
      <td>302461</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>23.731533</td>
      <td>150.780095</td>
      <td>977007748724</td>
      <td>52652291654</td>
      <td>283.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>751.0</td>
      <td>0.0</td>
      <td>43</td>
      <td>12981</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2010-05-24 15:00:00</th>
      <td>33811</td>
      <td>27383-WindGust-15-999</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>17.688117</td>
      <td>203.560972</td>
      <td>1278089510493</td>
      <td>101146546356</td>
      <td>278.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>899.0</td>
      <td>0.0</td>
      <td>55</td>
      <td>19045</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-18 12:00:00</th>
      <td>302448</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>23.322382</td>
      <td>159.561444</td>
      <td>909102411120</td>
      <td>73606522730</td>
      <td>282.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>859.0</td>
      <td>0.0</td>
      <td>58</td>
      <td>20265</td>
      <td>13241</td>
      <td>160548</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2013-10-13 07:00:00</th>
      <td>357719</td>
      <td>357345-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>96.950985</td>
      <td>328.960855</td>
      <td>913899151504</td>
      <td>101329529165</td>
      <td>270.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1350.0</td>
      <td>0.0</td>
      <td>60</td>
      <td>22637</td>
      <td>5111.0</td>
      <td>57971.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2011-06-01 12:00:00</th>
      <td>91295</td>
      <td>90882-WindGust-15-999</td>
      <td>WindGust</td>
      <td>15</td>
      <td>999</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>104964291709</td>
      <td>104964291709</td>
      <td>290.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1812.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>22328</td>
      <td>4202.0</td>
      <td>49336.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-18 11:00:00</th>
      <td>302442</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>25.676517</td>
      <td>146.906255</td>
      <td>835495888390</td>
      <td>29279167082</td>
      <td>282.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>830.0</td>
      <td>0.0</td>
      <td>83</td>
      <td>27791</td>
      <td>12946</td>
      <td>156366</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012-07-18 10:00:00</th>
      <td>302435</td>
      <td>NULL</td>
      <td>Pressure</td>
      <td>0</td>
      <td>1000</td>
      <td>-999.000000</td>
      <td>347.163683</td>
      <td>806216721308</td>
      <td>806216721308</td>
      <td>282.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>835.0</td>
      <td>0.0</td>
      <td>114</td>
      <td>29915</td>
      <td>11595</td>
      <td>139115</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>24557 rows × 51 columns</p>
</div>



#### Save/load dataset


```python
save_dataset(dataset)
if 'point_in_time' in dataset.columns:
    dataset.set_index('point_in_time', inplace=True) # This is called also inside load_dataset
```

    Storing classification set to db sasse.classification_dataset...


Load ready dataset from database:


```python
dataset = load_dataset()
```

### Energiateollisuus

#### Create dataset

Get dataset from db. This is a long process. If the dataset is already saved, this may be skipped.


```python
def get_ene_dataset(start, end, meta_params, storm_params, outage_params, all_params, paraller=True):
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

    print('.', end='')
    
    df = pd.DataFrame(results, columns=all_params)
    
    return df    
```


```python
%%time
from dask.distributed import progress
starttime = datetime.datetime.strptime('2009-12-31', "%Y-%m-%d")
endtime = datetime.datetime.strptime('2019-02-01', "%Y-%m-%d")
paraller = True

print('Reading data for {}-{}'.format(starttime, endtime))
# Progress bar
max_count = 108 # would be better to calculate
f = IntProgress(min=0, max=max_count) # instantiate the bar
display(f) # display the bar

dfs, df = [], []
start = starttime
while start <= endtime:
    end = start + timedelta(days=1)
    if paraller: dfs.append(delayed(get_ene_dataset)(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, storm_params, outage_params, all_params))
    else: df.append(get_ene_dataset(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), meta_params, storm_params, outage_params, all_params, paraller))
    start = end

if paraller:
    df = dask.compute(*dfs)
    progress(df)

ene_dataset = pd.concat(df)
    
ene_dataset.sort_values(by=['point_in_time'], inplace=True)
print('\nDone. Found {} records'.format(ene_dataset.shape[0]))
```

    Reading data for 2009-12-31 00:00:00-2019-02-01 00:00:00



    IntProgress(value=0, max=108)


If outages and customers are none, they are 0.


```python
ene_dataset.loc[:,['outages','customers']] = ene_dataset.loc[:,['outages','customers']].fillna(0)
ene_dataset.loc[:,['outages','customers']] = ene_dataset.loc[:,['outages','customers']].astype(int)

# Drop rows with missing meteorological params
for p in met_params:
    ene_dataset = ene_dataset[ene_dataset[p] != -999]

ene_dataset.sort_values(by=['outages'], inplace=True)
```


```python
ene_dataset.shape
```




    (142876, 48)



#### Cast classes

Following plots help us to divide data to classes.


```python
# Per transformer
fig, ax = plt.subplots(figsize=(20,5))
ene_dataset.loc[(ene_dataset['outages'] >= 1), :].groupby(by=['outages']).count()['id'].plot(kind='bar', title='Number of polygons by outages')
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

plt.ylabel('Number of polygons')
plt.grid(False)

# Per customer
fig, ax = plt.subplots(figsize=(20,5))
ene_dataset.loc[(ene_dataset['customers'] >= 1), :].groupby(by=['customers']).count()['id'].plot(kind='bar', title='Number of polygons by customers')
xtick_count = 15
every_nth = len(ax.xaxis.get_ticklabels()) // xtick_count
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
        
plt.ylabel('Number of polygons')
plt.grid(False)

# Scatter
fig, ax = plt.subplots(figsize=(5,5))
ene_dataset.plot(kind='scatter', x='outages', y='customers', ax=ax, c='green')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faaa057d9b0>




![png](output_42_1.png)



![png](output_42_2.png)



![png](output_42_3.png)


We can also scatter number of polygons against customers and outages.


```python
# Scatter
fig, ax = plt.subplots(figsize=(5,5))
df = pd.DataFrame(ene_dataset)
df['nro_polygons'] = df.loc[(df['outages'] >= 1), :].groupby(by=['outages']).count()['id']
df.plot(kind='scatter', x='nro_polygons', y='customers', ax=ax, c='green')
fig, ax = plt.subplots(figsize=(5,5))
df.plot(kind='scatter', x='nro_polygons', y='outages', ax=ax, c='green')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faaa98f09e8>




![png](output_44_1.png)



![png](output_44_2.png)


First, if number of outages is 0, the class is 0. Rest of the dataset could be divided to classes following (based on the first ploit): 

Class 1:   1 - 250  outages<br>
Class 2: 251 - 750  outages<br>
Class 3: 751 -      outages<br>

We can also classify the data based on number of customers respectively (based on the second plot): 

Class 1:    1 - 1000 customers<br>
Class 2: 1001 - 3000 customers<br>
Class 3: 3001 -      customers <br>


```python
# outages
limits = [(0,0), (1,250), (251,750), (751, 9999999)]
i = 0
for low, high in limits:
    ene_dataset.loc[(ene_dataset.loc[:, 'outages'] >= low) & (ene_dataset.loc[:, 'outages'] <= high), 'class'] = i
    i += 1
    
# outages
limits = [(0,1), (1,1000), (1001,3000), (3001, 9999999)]
i = 0
for low, high in limits:
    ene_dataset.loc[(ene_dataset.loc[:, 'customers'] >= low) & (ene_dataset.loc[:, 'customers'] <= high), 'class_customers'] = i
    i += 1

ene_dataset.loc[:, ['class', 'class_customers']] = ene_dataset.loc[:, ['class', 'class_customers']].astype(int)
print(ene_dataset.head())
print(ene_dataset.shape)
```

                                   id                storm_id weather_parameter  \
    point_in_time                                                                 
    2010-01-01 00:00:00+00:00    4576    4576-WindGust-15-999          WindGust   
    2014-10-23 00:00:00+00:00  482137  481833-WindGust-15-999          WindGust   
    2014-10-22 23:00:00+00:00  482083  481833-WindGust-15-999          WindGust   
    2014-10-22 23:00:00+00:00  482082  481465-WindGust-15-999          WindGust   
    2014-10-22 23:00:00+00:00  482099  478047-WindGust-15-999          WindGust   
    
                              low_limit high_limit  speed_self  angle_self  \
    point_in_time                                                            
    2010-01-01 00:00:00+00:00        15        999 -999.000000 -999.000000   
    2014-10-23 00:00:00+00:00        15        999    2.393847  336.148021   
    2014-10-22 23:00:00+00:00        15        999    3.264672  165.229990   
    2014-10-22 23:00:00+00:00        15        999    3.069429  172.936362   
    2014-10-22 23:00:00+00:00        20        999   10.211669   14.969946   
    
                                    area_m2    area_diff MIN Temperature  ...  \
    point_in_time                                                         ...   
    2010-01-01 00:00:00+00:00     282711750         -999             267  ...   
    2014-10-23 00:00:00+00:00   62408552814   2425513709             271  ...   
    2014-10-22 23:00:00+00:00   59983039105     92091335             272  ...   
    2014-10-22 23:00:00+00:00  745270279353  47217898519             265  ...   
    2014-10-22 23:00:00+00:00    9187397687  -2551154565             267  ...   
    
                               MAX Wind gust MAX Temperature SUM Snowfall  \
    point_in_time                                                           
    2010-01-01 00:00:00+00:00             15             267            0   
    2014-10-23 00:00:00+00:00             17             278            0   
    2014-10-22 23:00:00+00:00             17             278            0   
    2014-10-22 23:00:00+00:00             28             285            0   
    2014-10-22 23:00:00+00:00             22             272            0   
    
                              AVG Snowdepth AVG Mixed layer height  \
    point_in_time                                                    
    2010-01-01 00:00:00+00:00             0                    990   
    2014-10-23 00:00:00+00:00             0                    921   
    2014-10-22 23:00:00+00:00             0                    903   
    2014-10-22 23:00:00+00:00             0                   1010   
    2014-10-22 23:00:00+00:00             0                    753   
    
                              MAX Precipitation kg/m2 outages customers class  \
    point_in_time                                                               
    2010-01-01 00:00:00+00:00                       0       0         0     0   
    2014-10-23 00:00:00+00:00                       0       0         0     0   
    2014-10-22 23:00:00+00:00                       0       0         0     0   
    2014-10-22 23:00:00+00:00                       0       0         0     0   
    2014-10-22 23:00:00+00:00                       0       0         0     0   
    
                              class_customers  
    point_in_time                              
    2010-01-01 00:00:00+00:00               0  
    2014-10-23 00:00:00+00:00               0  
    2014-10-22 23:00:00+00:00               0  
    2014-10-22 23:00:00+00:00               0  
    2014-10-22 23:00:00+00:00               0  
    
    [5 rows x 49 columns]
    (142876, 49)


#### Save/load dataset


```python
save_dataset(ene_dataset, table_name='classification_dataset_energiateollisuus')
if 'point_in_time' in ene_dataset.columns:
    ene_dataset.set_index('point_in_time', inplace=True) # This is called also inside load_dataset
```

    Storing classification set to db sasse.classification_dataset_energiateollisuus...


Load ready dataset from database:


```python
ene_dataset = load_dataset(table_name='classification_dataset_energiateollisuus')
```

## Analysis

### Loiste & JSE


```python
df = dataset.copy()
```

#### Number of different types of polygons

Number of different types of polygons:


```python
df.groupby(by=['weather_parameter', 'low_limit']).count()['id'].plot(kind='bar', title='Number of polygons')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faab6ac72e8>




![png](output_56_1.png)



```python
df.groupby(by=['weather_parameter', 'low_limit']).count()['id']
```




    weather_parameter  low_limit
    Pressure           0            17880
    WindGust           15            6333
                       20             336
                       25               8
    Name: id, dtype: int64



#### Histogram of outage count

Number of outages caused by the storms.


```python
df.groupby(by=['outages']).count()['id']
```




    outages
    0      22597
    1       1207
    2        311
    3        139
    4         79
    5         40
    6         42
    7         36
    8         17
    9         13
    10        11
    11        13
    12         7
    13         4
    14         6
    15         2
    16         4
    17         2
    18         1
    19         2
    20         1
    23         2
    26         1
    27         1
    28         1
    32         1
    33         2
    34         1
    37         1
    39         1
    40         1
    41         1
    42         3
    43         1
    55         1
    58         1
    60         1
    75         1
    83         1
    114        1
    Name: id, dtype: int64




```python
fig, ax = plt.subplots(figsize=(20,5))
df.groupby(by=['outages']).count()['id'].plot(kind='bar', title='Number of outages', ax=ax)
plt.yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
```


![png](output_61_0.png)


#### Class sizes


```python
fig, ax = plt.subplots(figsize=(20,5))
df.groupby(by=['class']).count()['id'].plot(kind='bar', title='Number of polygon by classes', ax=ax)
plt.yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)

fig, ax = plt.subplots(figsize=(20,5))
df.groupby(by=['class_customers']).count()['id'].plot(kind='bar', title='Number of polygon by customer classes', ax=ax)
plt.yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
```


![png](output_63_0.png)



![png](output_63_1.png)


#### Timeseries of storms and outages

Storm objects cause some outages. It can be seen from a timeseries of storm objects and outages as well.


```python
def plot_outages(df, parameter, limit):

    fig, ax = plt.subplots(figsize=(15,6))
    ax2 = ax.twinx()
    df[(df.loc[:,'weather_parameter'] == parameter) & (df.loc[:,'low_limit'] == limit)].loc[:,'weather_parameter'].groupby(by=[pd.Grouper(freq='D')]).count().plot(ax=ax, label='Storm objects')
    df.loc[:, 'outages'].groupby(by=[pd.Grouper(freq='D')]).sum().plot(ax=ax2, color='r', linestyle='dashed', label='Outages')
    fig.autofmt_xdate()

    ax.set_ylabel('Number of storm objects')
    ax2.set_ylabel('Number of outages')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left')            
    
    plt.title('{} limit {}'.format(parameter, limit))
    
plot_outages(df, 'WindGust', 15)
plot_outages(df, 'WindGust', 20)
plot_outages(df, 'WindGust', 25)
```


![png](output_66_0.png)



![png](output_66_1.png)



![png](output_66_2.png)


#### Correlation between storms and outages

Correlation between storm objects and number of outages. Strong correlation exists.


```python
def plot_pval(corr_df, ax):

    coeffmat = np.zeros((corr_df.shape[1], corr_df.shape[1]))
    pvalmat = np.zeros((corr_df.shape[1], corr_df.shape[1]))

    for i in range(corr_df.shape[1]):
        for j in range(corr_df.shape[1]):        
            corrtest = pearsonr(corr_df[corr_df.columns[i]], corr_df[corr_df.columns[j]])  

            coeffmat[i,j] = corrtest[0]
            pvalmat[i,j] = corrtest[1]

    pvalmat_df = pd.DataFrame(pvalmat, columns=corr_df.columns, index=corr_df.columns)
    #fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(pvalmat_df, cmap="YlGnBu", annot=True, ax=ax)
    
def plot_corr(df):
    s = df[(df.loc[:,'weather_parameter'] == 'Pressure')].loc[:,'weather_parameter'].groupby(by=[pd.Grouper(freq='D')]).count()
    s2 = df[(df.loc[:,'weather_parameter'] == 'WindGust') & (df.loc[:,'low_limit'] == 15)].loc[:,'weather_parameter'].groupby(by=[pd.Grouper(freq='D')]).count()
    s3 = df[(df.loc[:,'weather_parameter'] == 'WindGust') & (df.loc[:,'low_limit'] == 20)].loc[:,'weather_parameter'].groupby(by=[pd.Grouper(freq='D')]).count()
    s4 = df[(df.loc[:,'weather_parameter'] == 'WindGust') & (df.loc[:,'low_limit'] == 25)].loc[:,'weather_parameter'].groupby(by=[pd.Grouper(freq='D')]).count()
    s5 = df.loc[:, 'outages'].groupby(by=[pd.Grouper(freq='D')]).sum()
    corr_df = pd.concat([s, s2, s3, s4, s5], axis=1, 
                        keys = ['Pressure', 'Wind Gust 15 m/s', 'Wind Gust 20 m/s', 'Wind Gust 25 m/s', 'Nro Outages'])
    corr_df.fillna(0, inplace=True)
    corr = corr_df.corr('pearson')

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    sns.heatmap(corr, cmap="YlGnBu", annot=True, ax=axes[0])
    axes[0].set_title('Correlation matrix')
    
    plot_pval(corr_df, ax=axes[1])
    axes[1].set_title('P-value test')
    plt.tight_layout()
```


```python
plot_corr(df)
```


![png](output_70_0.png)


#### Correlation between weather parameters and outages

Correlation between outages and different weather parameters (see two last rows). Significant correlation exists with at least: speed, area, temperature, standard deviation of the wind speed, total column water vapor, mixed layer height, CAPE, Dewpoint, total cloud cover, wind gust and pressure.

At least CAPE is most probably related to convective weather. 

P-value test passes for all parameters.


```python
def plot_pval_(corr_df, ax):

    coeffmat = np.zeros((corr_df.shape[1], corr_df.shape[1]))
    pvalmat = np.zeros((corr_df.shape[1], corr_df.shape[1]))

    for i in range(corr_df.shape[1]):
        for j in range(corr_df.shape[1]):        
            corrtest = pearsonr(corr_df[corr_df.columns[i]], corr_df[corr_df.columns[j]])  

            coeffmat[i,j] = corrtest[0]
            pvalmat[i,j] = corrtest[1]

    pvalmat_df = pd.DataFrame(pvalmat, columns=corr_df.columns, index=corr_df.columns)
    g = sns.heatmap(pvalmat_df, cmap="YlGnBu", annot=True, ax=ax, annot_kws={"size": 20}, cbar=False)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=45)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=45, rotation=45)
    
df2 = df.loc[:, storm_params + outage_params].fillna(0)
corr = df2.corr('pearson')

#fig, axes = plt.subplots(1, 2, figsize=(45,30))
fig, axes = plt.subplots(2, 1, figsize=(60,120))
g = sns.heatmap(corr, cmap="YlGnBu", annot=True, ax=axes[0], annot_kws={"size": 20}, cbar=False)
g.set_xticklabels(g.get_xmajorticklabels(), fontsize=45)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize=45, rotation=45)
axes[0].set_title('Correlation matrix')

plot_pval_(df2, ax=axes[1])
axes[1].set_title('P-value test')
plt.tight_layout()
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/scipy/stats/stats.py:3010: RuntimeWarning: invalid value encountered in double_scalars
      r = r_num / r_den



![png](output_73_1.png)


Let's consider only winter months (from beginning of October to end of January). Now we don't have significant correlation between CAPE and outages but we do have much stronger correlation between most other parameters. Distance to pressure object and wind speed introduce also a significant correlation with outages.

P-value test do not pass for several parameter any more, however. We have much less data (826 rows).


```python
df3 = df.loc[(df.index.month >= 10) | (df.index.month <= 1), storm_params + outage_params].fillna(0)
corr = df3.corr('pearson')

#fig, axes = plt.subplots(1, 2, figsize=(45,30))
fig, axes = plt.subplots(2, 1, figsize=(60,120))
g = sns.heatmap(corr, cmap="YlGnBu", annot=True, ax=axes[0], annot_kws={"size": 20}, cbar=False)
g.set_xticklabels(g.get_xmajorticklabels(), fontsize=45)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize=45, rotation=45)
axes[0].set_title('Correlation matrix')

plot_pval_(df3, ax=axes[1])
axes[1].set_title('P-value test')
plt.tight_layout()
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/scipy/stats/stats.py:3010: RuntimeWarning: invalid value encountered in double_scalars
      r = r_num / r_den



![png](output_75_1.png)


#### Scatter plots of outages and weather parameters

Scatter plot for outages with each parameters:


```python
def scatter_outages(df):
    fig, axes = plt.subplots(14,3, figsize=(30,90))
    row, col, i = 0, 0, 0
    for p in storm_params:
        row = int(i/3)
        col = i%3
        tdf = df.loc[(df[p] != -999), ['outages', p]].astype(float)
        tdf.plot(kind='scatter', x='outages', y=p, c='g', ax=axes[row][col])
        i += 1
```


```python
scatter_outages(df)
```


![png](output_79_0.png)


And the same for the winter months:


```python
scatter_outages(df3)
```


![png](output_81_0.png)


### Energiateollisuus


```python
df = ene_dataset.copy()
```

#### Number of different types of polygons


```python
df.groupby(by=['weather_parameter', 'low_limit']).count()['id'].plot(kind='bar', title='Number of polygons')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa78442828>




![png](output_85_1.png)



```python
df.groupby(by=['weather_parameter', 'low_limit']).count()['id']
```




    weather_parameter  low_limit
    Pressure           0            29690
    WindGust           15           93720
                       20           17240
                       25            1988
                       30             238
    Name: id, dtype: int64



#### Histogram of outage count

Number of outages caused by the storms.


```python
print(df.groupby(by=['outages']).count()['id'].head())
print('[...]')
print(df.groupby(by=['outages']).count()['id'].tail())
```

    outages
    0    107928
    1      3259
    2      4656
    3      1173
    4      2194
    Name: id, dtype: int64
    [...]
    outages
    27113    1
    27205    2
    27374    2
    28403    2
    28693    3
    Name: id, dtype: int64



```python
fig, ax = plt.subplots(figsize=(20,5))
df.groupby(by=['outages']).count()['id'].plot(kind='bar', title='Number of outages', ax=ax)
plt.yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
```


![png](output_90_0.png)


#### Class sizes


```python
fig, ax = plt.subplots(figsize=(20,5))
df.groupby(by=['class']).count()['id'].plot(kind='bar', title='Number of polygon by classes', ax=ax)
plt.yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)

fig, ax = plt.subplots(figsize=(20,5))
df.groupby(by=['class_customers']).count()['id'].plot(kind='bar', title='Number of polygon by customer classes', ax=ax)
plt.yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
```


![png](output_92_0.png)



![png](output_92_1.png)


#### Timeseries of storms and outages

Storm objects cause some outages. It can be seen from a timeseries of storm objects and outages as well.


```python
def plot_outages(df, parameter, limit):

    fig, ax = plt.subplots(figsize=(15,6))
    ax2 = ax.twinx()
    df[(df.loc[:,'weather_parameter'] == parameter) & (df.loc[:,'low_limit'] == limit)].loc[:,'weather_parameter'].groupby(by=[pd.Grouper(freq='D')]).count().plot(ax=ax, label='Storm objects')
    df.loc[:, 'outages'].groupby(by=[pd.Grouper(freq='D')]).sum().plot(ax=ax2, color='r', linestyle='dashed', label='Outages')
    fig.autofmt_xdate()

    ax.set_ylabel('Number of storm objects')
    ax2.set_ylabel('Number of outages')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left')            
    
    plt.title('{} limit {}'.format(parameter, limit))
    
plot_outages(df, 'WindGust', 15)
plot_outages(df, 'WindGust', 20)
plot_outages(df, 'WindGust', 25)
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/arrays/datetimes.py:1172: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
      "will drop timezone information.", UserWarning)
    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/arrays/datetimes.py:1172: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
      "will drop timezone information.", UserWarning)
    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/arrays/datetimes.py:1172: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
      "will drop timezone information.", UserWarning)
    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/arrays/datetimes.py:1172: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
      "will drop timezone information.", UserWarning)
    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/arrays/datetimes.py:1172: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
      "will drop timezone information.", UserWarning)
    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/arrays/datetimes.py:1172: UserWarning: Converting to PeriodArray/Index representation will drop timezone information.
      "will drop timezone information.", UserWarning)



![png](output_95_1.png)



![png](output_95_2.png)



![png](output_95_3.png)


#### Correlation between storms and outages

Correlation between storm objects and number of outages. Strong correlation exists.


```python
plot_corr(df) # Function defined in Loiste & JSE analysis
```


![png](output_98_0.png)


#### Correlation between weather parameters and outages

Correlation between outages and different weather parameters (see two last rows). Significant correlation exists with at least: area_m2, Max wind speed, STD Wind Speed, Max Wind Gust, STD Wind Gust, Max Precipitation kg/m2, SUM Precipitation KG/m2, Sum Snowfall, AVG Mixed layer height, Max Mixed layer height.

P-value test passes for all parameters having significant correlation except for Max precipitation kg/m2 (why?).


```python
df2 = df.loc[:, storm_params + outage_params].fillna(0)
corr = df2.corr('pearson')

#fig, axes = plt.subplots(1, 2, figsize=(45,30))
fig, axes = plt.subplots(2, 1, figsize=(60,120))
g = sns.heatmap(corr, cmap="YlGnBu", annot=True, ax=axes[0], annot_kws={"size": 20}, cbar=False)
g.set_xticklabels(g.get_xmajorticklabels(), fontsize=45)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize=45, rotation=45)
axes[0].set_title('Correlation matrix')

plot_pval_(df2, ax=axes[1]) # Function defined in Loiste & JSE Analysis
axes[1].set_title('P-value test')
plt.tight_layout()
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/scipy/stats/stats.py:3010: RuntimeWarning: invalid value encountered in double_scalars
      r = r_num / r_den



![png](output_101_1.png)


Let's consider only winter months (from beginning of October to end of January). Correlations are same with whole dataset.

P-value test passes for all paratemers having significant correlation  except AVG windspeed (why?).


```python
df3 = df.loc[(df.index.month >= 10) | (df.index.month <= 1), storm_params + outage_params].fillna(0)
corr = df3.corr('pearson')

#fig, axes = plt.subplots(1, 2, figsize=(45,30))
fig, axes = plt.subplots(2, 1, figsize=(60,120))
g = sns.heatmap(corr, cmap="YlGnBu", annot=True, ax=axes[0], annot_kws={"size": 20}, cbar=False)
g.set_xticklabels(g.get_xmajorticklabels(), fontsize=45)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize=45, rotation=45)
axes[0].set_title('Correlation matrix')

plot_pval_(df3, ax=axes[1])
axes[1].set_title('P-value test')
plt.tight_layout()
```

    /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/scipy/stats/stats.py:3010: RuntimeWarning: invalid value encountered in double_scalars
      r = r_num / r_den



![png](output_103_1.png)


#### Scatter plots of outages and weather parameters


```python
scatter_outages(df) # Function defined in Loiste & JSE analysis
```


![png](output_105_0.png)


And the same for the winter months:


```python
scatter_outages(df3)
```


![png](output_107_0.png)

