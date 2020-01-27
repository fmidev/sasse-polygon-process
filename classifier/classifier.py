# -*- coding: utf-8 -*-
import sys, argparse, logging, yaml
sys.path.insert(0, 'lib/')
from dbhandler import DBHandler
from filehandler import FileHandler
import datetime as dt
from datetime import timedelta
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model.logical import Logical

def get_param_names(config_filename, shortnames=True):
    """ Get param names, partly from config and partly as hard coded """

    with open(config_filename) as f:
        file_content = f.read()

        config_dict = yaml.load(file_content, Loader=yaml.FullLoader)

        params = config_dict['params']
        met_params = set()
        for param, info in params.items():
            for f in info['aggregation']:
                if shortnames:
                    met_params.add(f[1:]+' '+info['name'])
                else:
                    met_params.add(f+'{'+param+'}')

    met_params = list(met_params)
    polygon_params = ['speed_self', 'angle_self', 'area_m2', 'area_diff', 'speed_pressure', 'angle_pressure', 'distance_to_pressure', 'low_limit']
    meta_params = ['id', 'storm_id', 'point_in_time', 'weather_parameter', 'high_limit']
    outage_params = ['outages', 'customers']
    transformers_params = ['transformers', 'all_customers']
    storm_params = polygon_params + met_params

    all_params = meta_params + storm_params + outage_params + transformers_params

    return met_params, meta_params, storm_params, outage_params, transformers_params, all_params

def classify(data):
    """
    Add classification based on number of outages
    """
    def c(x):
        if x > 10: return 3
        if x > 5: return 2
        if x > 1: return 1
        return 0

    data['class'] = data['outages'].apply(c)
    return data

def main():

    # Read in data and extract data arrays
    #logging.info("Reading input files.")

    dbh = DBHandler(options.db_config_filename, options.db_config_name)
    dbh.return_df = False

#    fh = FileHandler(s3_bucket='fmi-sasse-classification-dataset')

    starttime = dt.datetime.strptime(options.starttime, "%Y-%m-%dT%H:%M:%S")
    endtime = dt.datetime.strptime(options.endtime, "%Y-%m-%dT%H:%M:%S")

    met_params, meta_params, storm_params, outage_params, transformers_params, all_params = get_param_names(options.param_config_filename)

    data = pd.DataFrame(dbh.get_dataset(all_params), columns=all_params)
    data = classify(data)
    print(data)
    data = data.loc[data['weather_parameter'] == 'WindGust']
    data_train, data_test = train_test_split(data)

    X_train = data_train.loc[:, storm_params].values
    y_train = data_train.loc[:, 'class']. values

    X_test = data_test.loc[:, storm_params].values
    y_test = data_test.loc[:, 'class']. values

    model = Logical()
    y_pred = model.predict(data_test)

    print(y_test)
    print(y_pred)
    print(classification_report(y_pred, y_test))

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, default='2011-02-01T00:00:00', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2011-03-01T00:00:00', help='End time of the classification data interval')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--db_config_name', type=str, default='production', help='Section name in db cnf file to read connection parameters')
    parser.add_argument('--param_config_filename', type=str, default='cnf/smartmet.yaml', help='Param config filename')
    parser.add_argument('--dataset_file', type=str, default='data/dataset.csv', help='If set, read dataset from csv file')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    main()
