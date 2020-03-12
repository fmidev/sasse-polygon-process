# -*- coding: utf-8 -*-
import sys, argparse, logging
sys.path.insert(0, 'lib/')
import datetime as dt
from datetime import timedelta
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from dbhandler import DBHandler
from filehandler import FileHandler
from viz import Viz

from model.logical import Logical
from util import evaluate
from config import read_options

def main():

    fh = FileHandler() #s3_bucket='fmi-sasse-classification-dataset')
    viz = Viz()

    data = pd.read_csv(options.test_dataset_file)
    data = data.loc[data['weather_parameter'] == 'WindGust']

    X = data.loc[:, options.feature_params]
    y = data.loc[:, options.label].values.ravel()

    model = fh.load_model(options.save_path)
    evaluate(model, options, data=(X,y), fh=fh, viz=viz)

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

    main()
