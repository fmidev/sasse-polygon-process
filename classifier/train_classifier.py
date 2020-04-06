# -*- coding: utf-8 -*-
import sys, argparse, logging, joblib
sys.path.insert(0, 'lib/')
from dbhandler import DBHandler
from filehandler import FileHandler
import datetime as dt
from datetime import timedelta
import pandas as pd
from math import floor

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

from dask.distributed import Client, progress

from model.logical import Logical
from util import cv, evaluate, feature_selection
from config import read_options
from viz import Viz

def main():

    #if hasattr(options, 'dask'): client = Client('{}:8786'.format(options.dask))
    # else: client = Client()

    if hasattr(options, 's3_bucket'):
        fh = FileHandler(s3_bucket=options.s3_bucket)
        viz = Viz(io=fh)
    else:
        fh = FileHandler()
        viz = Viz()

    data = pd.read_csv(options.dataset_file)
    data = data.loc[data['weather_parameter'] == 'WindGust']

    if options.debug:
        c = min(floor(len(data)/2), 5000)
        y = data.loc[:, options.label].values.ravel()
        data_train, data_test, _, __ = train_test_split(data, y, train_size=c, test_size=c, stratify=y)
    else:
        data_train, data_test = train_test_split(data)

    X_train = data_train.loc[:, options.feature_params]
    y_train = data_train.loc[:, options.label].values.ravel()

    X_test = data_test.loc[:, options.feature_params]
    y_test = data_test.loc[:, options.label].values.ravel()

    # Normalise features
    if options.normalize:
        logging.info('Normalizing...')
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=options.feature_params)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=options.feature_params)
        fh.save_scaler(scaler, options.save_path)

    # SMOTE
    if options.smote:
        logging.info('Smoting...')
        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=options.feature_params)

    # Initialize model
    if options.model == 'Logical':
        model = Logical()
    elif options.model == 'rfc':
        model = RandomForestClassifier(n_jobs=-1)
    elif options.model == 'svc':
        model = SVC(gamma='scale')
    else:
        raise Exception("Model not implemented")

    # Run CV / Feature selection / train
    if options.cv:
        model = cv(X_train, y_train, model, options, fh)
    elif options.feature_selection:
        model, params = feature_selection(X_train, y_train, model, options, fh)
        logging.info('Best model got with params {}'.format(','.join(params)))
    else:
        logging.info('Traingin {} with {} samples...'.format(options.model, len(X_train)))
        #with joblib.parallel_backend('dask'):
        model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    logging.info('Training report:\n{}'.format(classification_report(y_pred_train, y_train)))

    #y_pred = model.predict(X_test)

    logging.info('Performance for validation data:')
    evaluate(model, options, data=(X_test, y_test), fh=fh, viz=viz)

    # logging.info('Validation report:\n{}'.format(classification_report(y_pred, y_test)))
    if hasattr(options, 'test_dataset'):
        logging.info('Performance for {}:'.format(options.test_dataset))
        evaluate(model, options, dataset_file=options.test_dataset, fh=fh, viz=viz)

    fh.save_model(model, options.save_path)

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

    main()
