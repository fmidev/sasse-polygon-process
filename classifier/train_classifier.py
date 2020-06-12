# -*- coding: utf-8 -*-
import sys, argparse, logging, joblib
sys.path.insert(0, 'lib/')
from filehandler import FileHandler
import pandas as pd
import numpy as np
from scipy.stats import expon

from dask.distributed import Client, progress
from sklearn.metrics import classification_report

from model.svct import SVCT

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, PairwiseKernel

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from util import evaluate, cv
from config import read_options
from viz import Viz

def main():

    if hasattr(options, 'dask'): client = Client('{}:8786'.format(options.dask))
    else: client = Client()

    logging.info(client)

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
    if options.model == 'svct':
        model = SVCT(verbose=True)
    elif options.model == 'gp':
        kernel = PairwiseKernel(metric='laplacian') *  DotProduct()
        model = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
    elif options.model == 'rfc':
        # param_grid_rfc = {
        # "n_estimators": [10, 100, 150, 200, 250, 500],
        # "max_depth": [20, 50, 100, None],
        # "max_features": ["auto", "log2", None],
        # "min_samples_split": [2,5,10],
        # "min_samples_leaf": [1, 2, 4],
        # "bootstrap": [False]
        # }

        # Fetched using 5-fold cv with random search from params above
        if "national" in options.dataset:
            params = {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False, 'n_jobs': -1}
        else:
            params = {'n_estimators': 250, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': None, 'max_depth': 20, 'bootstrap': False, 'n_jobs': -1}

        model = RandomForestClassifier(**params)

    elif options.model == 'gnb':
        model = GaussianNB()
    else:
        raise Exception('Model not defined')

    logging.info('Training...')
    if options.model == 'gnb':
        priors = []
        for i in np.arange(0,1,.05):
            for j in np.arange(0, 1-i, .05):
                k = 1 - i - j
                priors.append([i, j, k])

        param_grid_gnb = {
        'priors': priors+[None],
        'var_smoothing': expon(scale=.01)
        }
        model, cv_results = cv(model, param_grid_gnb, X_train, y_train, n_iter=500)
    else:
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

    if options.model == 'rfc':
        # Sort feature importances in descending order and rearrange feature names accordingly
        indices = np.argsort(model.feature_importances_)[::-1]
        names = [options.feature_params[i] for i in indices]
        importances = model.feature_importances_[indices]

        fname = '{}/feature_importances.png'.format(options.output_path)

        viz.rfc_feature_importance(importances, fname, names)

    fh.save_model(model, options.save_path)


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default='cnf/rfc.ini', help='Config filename for training config')
    parser.add_argument('--config_name', type=str, default='thin', help='Config section for training config')
    parser.add_argument('--train_data', type=str, default='', help='Train dtaset file')
    parser.add_argument('--test_data', type=str, default='', help='Test dataset file')
    parser.add_argument('--model', type=str, default='svct', help='Used model')
    parser.add_argument('--dataset', type=str, default='local_random', help='For saving results and model')
    parser.add_argument('--max_size', type=int, default=None, help='Cut data to this size')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()
    read_options(options)

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    logging.info('Using config {} from {}'.format(options.config_name, options.config_filename))

    main()
