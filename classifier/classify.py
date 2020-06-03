# -*- coding: utf-8 -*-
import sys, argparse, logging, yaml
sys.path.insert(0, 'lib/')
from dbhandler import DBHandler
from filehandler import FileHandler
import datetime as dt
from datetime import timedelta
import pandas as pd
from util import get_param_names
#from config import read_options

def main():

    # Read in data and extract data arrays
    logging.info("Reading input data for {} - {}...".format(options.starttime, options.endtime))

    #if options.dataset_file is None or options.output_file is None:
    dbh = DBHandler(options.db_config_filename, options.db_config_name)
    dbh.return_df = False
    fh = FileHandler(s3_bucket=options.bucket)

    starttime = dt.datetime.strptime(options.starttime, "%Y-%m-%dT%H:%M:%S")
    endtime = dt.datetime.strptime(options.endtime, "%Y-%m-%dT%H:%M:%S")

    features, meta_params, labels, all_params = get_param_names(options.param_config_filename)

    scaler = fh.load_model(options.scaler_file)

    # TODO change to read from operational data
    data = pd.DataFrame(dbh.get_dataset(all_params), columns=all_params)

    # TODO use original id stored in db. The id is used to assign predicted classes to a storm object (while saving to db)
    # As far as we do not have operational data, dummy ids are used
    data.loc[:, 'id'] = 0

    data = data.loc[data['weather_parameter'] == 'WindGust']

    # Add week
    data['point_in_time'] = pd.to_datetime(data['point_in_time'])
    data['week'] = data['point_in_time'].dt.week

    X = data.loc[:, features]
    X = scaler.transform(X)

    model = fh.load_model(options.model_file)

    logging.info('Predicting with {} {} samples...'.format(options.model, len(X)))
    y_pred = model.predict(X)

    # Save to db
    logging.info('Saving...')
    dbh.save_classes(data.loc[:, 'id'], y_pred)
    logging.info('done.')

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, default='2010-06-12T00:00:00', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2010-06-13T00:00:00', help='End time of the classification data interval')
    parser.add_argument('--db_config_filename', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--db_config_name', type=str, default='production', help='Section name in db cnf file to read connection parameters')
    parser.add_argument('--bucket', type=str, default='fmi-asi-sasse-assets', help='Bucket name where models are stored')
    parser.add_argument('--param_config_filename', type=str, default='cnf/smartmet.yaml', help='Param config filename')
    #parser.add_argument('--config_filename', type=str, default='cnf/options.ini', help='Config filename for training config')
    #parser.add_argument('--config_name', type=str, default='thin', help='Config section for training config')
    parser.add_argument('--model_file', type=str, default=None, help='Model filename (required)')
    parser.add_argument('--scaler_file', type=str, default=None, help='Scaler filename (required)')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()
    #read_options(options)

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    main()
