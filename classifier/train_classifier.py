# -*- coding: utf-8 -*-
import sys, argparse, logging
sys.path.insert(0, 'lib/')
from dbhandler import DBHandler
from filehandler import FileHandler
import datetime as dt
from datetime import timedelta
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model.logical import Logical
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

    data = pd.DataFrame(dbh.get_dataset(all_params), columns=all_params)
    #data = classify(data)
    #print(data)
    data = data.loc[data['weather_parameter'] == 'WindGust']
    data_train, data_test = train_test_split(data)

    X_train = data_train.loc[:, features]
    y_train = data_train.loc[:, options.label]

    X_test = data_test.loc[:, features]
    y_test = data_test.loc[:, options.label]

    if options.model == 'Logical':
        model = Logical()
    elif options.model == 'RFC':
        model = RandomForestClassifier(n_jobs=-1)
    else:
        raise "Model not implemented"

    logging.info('Traingin {} with {} samples...'.format(options.model, len(X_train)))
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    logging.info('Training report:\n{}'.format(classification_report(y_pred_train, y_train)))

    y_pred = model.predict(X_test)

    logging.info('Validatio report:\n{}'.format(classification_report(y_pred, y_test)))

    model_savepath = get_savepath(options)
    fh.save_model(model, model_savepath)

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
