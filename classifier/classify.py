# -*- coding: utf-8 -*-
import sys, argparse, logging, yaml
sys.path.insert(0, 'lib/')
from dbhandler import DBHandler
from filehandler import FileHandler
import datetime as dt
from datetime import timedelta
import pandas as pd
from util import get_param_names, get_savepath


def main():

    # Read in data and extract data arrays
    #logging.info("Reading input files.")

    dbh = DBHandler(options.db_config_filename, options.db_config_name)
    dbh.return_df = False
    fh = FileHandler() #s3_bucket='fmi-sasse-classification-dataset')

    starttime = dt.datetime.strptime(options.starttime, "%Y-%m-%dT%H:%M:%S")
    endtime = dt.datetime.strptime(options.endtime, "%Y-%m-%dT%H:%M:%S")

    features, meta_params, lables, all_params = get_param_names(options.param_config_filename)

    # TODO change to read from operational data
    data = pd.DataFrame(dbh.get_dataset(all_params), columns=all_params)
    # As far as we do not have operational data, dummy ids are used
    data.loc[:, 'id'] = 0

    data = data.loc[data['weather_parameter'] == 'WindGust']
    X = data.loc[:, features]

    model_savepath = get_savepath(options)
    model = fh.load_model(model_savepath)

    logging.info('Predicting with {} {} samples...'.format(options.model, len(X)))
    y_pred = model.predict(X)

    # Save to db
    dbh.save_classes(data.loc[:, 'id'], y_pred)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, default='2011-02-01T00:00:00', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2011-03-01T00:00:00', help='End time of the classification data interval')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--db_config_name', type=str, default='production', help='Section name in db cnf file to read connection parameters')
    parser.add_argument('--param_config_filename', type=str, default='cnf/smartmet.yaml', help='Param config filename')
    parser.add_argument('--dataset_file', type=str, default='data/classification_dataset_loiste_jse.csv', help='If set, read dataset from csv file')
    parser.add_argument('--model_savepath', type=str, default='models', help='If set, read dataset from csv file')
    parser.add_argument('--model', type=str, default='RFC', help='Classifier')
    parser.add_argument('--label', type=str, default='class', help='Classifier')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    main()
