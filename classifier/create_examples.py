# -*- coding: utf-8 -*-
import sys, argparse, logging, yaml
sys.path.insert(0, 'lib/')
from dbhandler import DBHandler
from filehandler import FileHandler
import datetime as dt
from datetime import timedelta
import pandas as pd
from util import get_param_names
from config import read_options

def main():

    #if options.dataset_file is None or options.output_file is None:
    dbh = DBHandler(options.db_config_file, options.db_config_name)
    dbh.return_df = False
    fh = FileHandler(s3_bucket=options.bucket)

    with open(options.example_config_file) as f:
        setups = yaml.load(f.read(), Loader=yaml.FullLoader)

    for setup in setups['examples']:

        output_file = '{}/{}-{}.csv'.format(setup['output_dir'],
                                            setup['starttime'].strftime('%Y%m%dT%H%M%S'),
                                            setup['endtime'].strftime('%Y%m%dT%H%M%S'))

        # Read in data and extract data arrays
        logging.info("Reading input data for {} - {}...".format(setup['starttime'], setup['endtime']))

        features, meta_params, labels, all_params = get_param_names(options.param_config_file)

        data = fh.read_data([setup['dataset_file']],
                            options,
                            return_meta=True,
                            starttime=setup['starttime'].strftime('%Y-%m-%dT%H:%M:%S'),
                            endtime=setup['endtime'].strftime('%Y-%m-%dT%H:%M:%S'))[0]
        X, y, meta = data

        model = fh.load_model(setup['model_file'])
        scaler = fh.load_model(setup['scaler_file'])
        
        logging.info('Predicting with {} samples...'.format(len(X)))
        y_pred = model.predict(X)

        df = pd.DataFrame(meta, columns=options.meta_params)
        X_inv = pd.DataFrame(scaler.inverse_transform(X), columns=X.columns)
        df = pd.concat([df.reset_index(drop=True), X_inv.reset_index(drop=True)], axis=1)
        df = dbh.get_geom_for_dataset_rows(df)
        df['y_pred'] = y_pred
        df['y'] = y
        fh.df_to_csv(df, output_file)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--example_config_file', type=str, default='cnf/examples.yaml', help='Example setup filename')
    parser.add_argument('--db_config_file', type=str, default='cnf/sasse_aws.yaml', help='CNF file containing DB connection pararemters')
    parser.add_argument('--db_config_name', type=str, default='production', help='Section name in db cnf file to read connection parameters')
    parser.add_argument('--bucket', type=str, default='fmi-asi-sasse-assets', help='Bucket name where models are stored')
    parser.add_argument('--param_config_file', type=str, default='cnf/smartmet.yaml', help='Param config filename')
    parser.add_argument('--config_filename', type=str, default='cnf/options.ini', help='Config filename for training config')
    parser.add_argument('--config_name', type=str, default='thin', help='Config section for training config')
    parser.add_argument('--output_file', type=str, default=None, help='If set, results are saved to given csv file')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    options = parser.parse_args()
    read_options(options)

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"),
                        level=logging.INFO)

    main()
