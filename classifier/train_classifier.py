# -*- coding: utf-8 -*-
import sys, argparse, logging, joblib
sys.path.insert(0, 'lib/')
from filehandler import FileHandler
import pandas as pd
import numpy as np

from dask.distributed import Client, progress
from sklearn.metrics import classification_report

from model.svct import SVCT
from util import evaluate
from config import read_options
from viz import Viz

def main():

    if hasattr(options, 'dask'): client = Client('{}:8786'.format(options.dask))
    else: client = Client()

    if hasattr(options, 's3_bucket'):
        fh = FileHandler(s3_bucket=options.s3_bucket)
        viz = Viz(io=fh)
    else:
        fh = FileHandler()
        viz = Viz()

    datasets = fh.read_data([options.train_data, options.test_data], options)

    X_train = datasets[0][0]
    y_train = datasets[0][1]
    X_test = datasets[1][0]
    y_test = datasets[1][1]

    # Train
    model = SVCT(verbose=True)

    with joblib.parallel_backend('dask'):
        model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    logging.info('Training report:\n{}'.format(classification_report(y_train, y_pred_train)))

    y_pred = model.predict(X_test)
    logging.info('Validation report:\n{}'.format(classification_report(y_test, y_pred)))

    fname = '{}/confusion_matrix_testset.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(3), filename=fname)

    fname = '{}/confusion_matrix_testset_normalised.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(3), True, filename=fname)

    fh.save_model(model, options.save_path)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default='cnf/rfc.ini', help='Config filename for training config')
    parser.add_argument('--config_name', type=str, default='thin', help='Config section for training config')
    parser.add_argument('--train_data', type=str, default='', help='Train dtaset file')
    parser.add_argument('--test_data', type=str, default='', help='Test dataset file')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()
    read_options(options)

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    logging.info('Using config {} from {}'.format(options.config_name, options.config_filename))

    main()
