# -*- coding: utf-8 -*-
"""
CSV file handler
"""
import sys, logging, os, boto3
from pathlib import Path
import numpy as np
#import datetime
#from configparser import ConfigParser
#import pandas.io.sql as sqlio
import pandas as pd
import geopandas as gpd
from shapely import wkt
from joblib import dump, load
from model.svct import SVCT

class FileHandler(object):

    s3 = False
    gs = False
    bucket_name = ''
    bucket = ''
    client = ''

    """
    Create, store and read csv files
    """
    def __init__(self, s3_bucket=None):
        # self.config_filename = config_filename
        # self.config_name = config_name
        if s3_bucket is not None:
            self.bucket_name = s3_bucket
            self.s3 = True
            self.client = boto3.client('s3')
            resource = boto3.resource('s3')
            self.bucket = resource.Bucket(self.bucket_name)

    def cut_classes(self, dataset, classes, max_size, label):
        """ Decrease dataset size by cutting requested classes smaller """

        # Cherry picked classes
        class_dfs = []
        for c in classes:
            picked_data = dataset.loc[(dataset.loc[:,label] == c),:].reset_index(drop=True)
            class_dfs.append(picked_data.loc[0:min(len(picked_data), max_size),:])
            #class_dfs.append(picked_data.sample(n=min(len(picked_data), max_size)))

        # Concat
        data = pd.concat(class_dfs)
        return data

    def read_data(self, filenames, options, return_meta=False, starttime=None, endtime=None):
        """
        Read data from csv file(s). Download it from bucket if necessary
        """
        datasets = []

        for f in filenames:
            self._download_from_bucket(f, f)

            # Train
            data = pd.read_csv(f)

            missing = list(set(options.feature_params + options.meta_params + options.label)-set(data.columns.values))
            if len(missing) > 0:
                logging.warning("Missing parameter(s) {}".format(','.join(missing)))

            if starttime is not None or endtime is not None:
                data['point_in_time'] = pd.to_datetime(data['point_in_time'], utc=True)
            if starttime is not None:
                data = data[(data['point_in_time'] >= starttime)]
            if endtime is not None:
                data = data[(data['point_in_time'] <= endtime)]

            if options.max_size is not None:
                data = self.cut_classes(data, [0,1,2], options.max_size, options.label[0])

            X = data.loc[:, options.feature_params]
            y = data.loc[:, options.label].values.ravel()

            meta = None
            if return_meta:
                meta = data.loc[:, options.meta_params]

            logging.info('Read data from {}, shape: {}'.format(f, X.shape))

            datasets.append((X, y, meta))

        return datasets

    def dataset_from_csv(self, filename, time_column='point_in_time'):
        """
        Read dataset from csv file

        filename : str
                   filename to use

        return DataFrame
        """
        return pd.from_csv(filename, parse_dates=[time_column])

    def load_model(self, filename, force=False):
        """ Load model from given path """
        logging.info('Loading model from {}...'.format(filename))
        self._download_from_bucket(filename, filename, force=force)
        return load(filename)

    def load_svct(self, save_path, force=False):
        """
        Load SVCT
        """
        logging.info('Loading model from {}...'.format(save_path))
        fname1 = save_path + '/model1.joblib'
        self._download_from_bucket(fname1, fname1, force=force)
        model1 = load(fname1)

        fname2 = save_path + '/model2.joblib'
        self._download_from_bucket(fname2, fname2, force=force)
        model2 = load(fname2)

        model = SVCT()
        model.model1 = model1
        model.model2 = model2

        return model

    def save_svct(self, model, save_path):
        """
        Save SVCT
        """
        fname1 = save_path + '/model1.joblib'
        self.save_scikit_file(model.model1, fname1)
        fname2 = save_path + '/model2.joblib'
        self.save_scikit_file(model.model2, fname2)

    def save_model(self, model, save_path):
        """
        Save model to given path
        """
        fname = save_path + '/model.joblib'
        self.save_scikit_file(model, fname)

    def save_scaler(self, scaler, save_path):
        """ Save scikit scaler """

        fname = save_path + '/scaler.joblib'
        self.save_scikit_file(scaler, fname)

    def save_scikit_file(self, model, fname):
        """ Save scikit file """

        save_path = os.path.dirname(os.path.abspath(fname))

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        dump(model, fname)
        logging.info('Saved to {}'.format(fname))

        if self.s3:
            self._upload_to_bucket(fname, fname)

    def df_to_csv(self, df, local_filename, store_header=True):
        """
        Store Pandas DataFrame to csv file and upload it to bucket if ext_filename is set
        """
        Path(os.path.dirname(local_filename)).mkdir(parents=True, exist_ok=True)
        if df is not None and len(df) > 0:
            df.to_csv(local_filename, header=store_header, index_label='id')
            logging.info('Stored data to {}'.format(local_filename))
        else:
            open(local_filename, 'a').close()
            logging.info('No data, created empty file {}'.format(local_filename))

        if self.s3:
            self._upload_to_bucket(local_filename, local_filename)

    def save_prediction(self, meta, y_pred, y, filename):
        """
        Save prediction results to csv file for visualisation purposes.
        """
        df = pd.DataFrame(meta)
        df['y_pred'] = y_pred
        df['y'] = y
        print(df)
        df.loc[:, 'id'] = df.index
        self.df_to_csv(df, filename, store_header=False)

    def _upload_dir_to_bucket(self, path, ext_path):
        """
        Upload all files from folder to bucket
        """
        for file in os.listdir(path):
            self._upload_to_bucket(path+'/'+file, ext_path+'/'+file)

    def _upload_to_bucket(self, filename, ext_filename):
        """
        Upload file to bucket if bucket is set and ext_filename is not None
        """
        if ext_filename is None:
            return

        if self.s3:
            self.bucket.upload_file(filename, ext_filename)
            logging.info('Uploaded {} to S3 with name {}'.format(filename, ext_filename))
        if self.gs:
            try:
                client = storage.Client()
                bucket = client.get_bucket(self.bucket_name)
                blob = storage.Blob(ext_filename, bucket)
                blob.upload_from_filename(filename)
                logging.info('Uploaded to {}'.format(ext_filename))
            except:
                logging.warning('Uploading file to bucket failed')

    def _download_dir_from_bucket(self, ext_path, local_path, force=False):
        """
        Download all files from bucket and save them to 'local_path'
        """
        if os.path.exists(local_path) and not force:
            logging.info('Path {} already exists. Not overwriting...'.format(local_path))
            return
        if os.path.exists(local_path) and force:
            logging.info('Path {} already exists. Overwriting...'.format(local_path))

        if self.s3:
             for object in self.bucket.objects.filter(Prefix = remoteDirectoryName):
                 local_name = object.key.replace(ext_path, local_path)
                 self._download_from_bucket(object.key, local_name)

        if self.gs:
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=ext_path)

            for blob in blobs:
                local_name = blob.name.replace(ext_path, local_path)
                self._download_from_bucket(blob.name, local_name, force)

    def _download_from_bucket(self, ext_filename, local_filename, force=False):
        """
        Download file from bucket and save it to 'local_filename'
        """
        if os.path.exists(local_filename) and not force:
            logging.info('File {} already exists. Not overwriting...'.format(local_filename))
            return
        if os.path.exists(local_filename) and force:
            logging.info('File {} already exists. Overwriting...'.format(local_filename))
        else:
            logging.info('File {} does not exist. Downloading...'.format(local_filename))

        Path(os.path.dirname(local_filename)).mkdir(parents=True, exist_ok=True)

        if self.s3:
            self.bucket.download_file(ext_filename, local_filename)
            logging.info('Downloaded {} to {}'.format(ext_filename, local_filename))
        if self.gs:
            try:
                client = storage.Client()
                bucket = client.get_bucket(self.bucket_name)
                blob = storage.Blob(ext_filename, bucket)
                blob.download_to_filename(local_filename)
                logging.info('Downloaded {} to {}'.format(ext_filename, local_filename))
            except:
                logging.warning('Downloading failed')

                i += 1

    def write_csv(self, _dict, filename):
        """
        Write dict to csv

        _dict : dict
                data in format ['key': [values], 'key2': values]
        filename : str
                   filename where data is saved
        """
        with open(filename, 'w') as f:
            f.write('"'+'";"'.join(_dict.keys())+'"\n')
            for i in np.arange(len(_dict[list(_dict.keys())[0]])):
                values = []
                for col in _dict.keys():
                    try:
                        values.append(str(_dict[col][i]))
                    except IndexError as e:
                        # LSTM don't have first times available because of lacking history
                        pass
                f.write(';'.join(values)+'\n')

        logging.info('Wrote {}'.format(filename))
        self._upload_to_bucket(filename, filename)

    def report_cv_results(self, results, scores=['score'], filename=None, n_top=5):
        """
        Report CV results and save them to file
        """
        res = ""
        for score in scores:

            res += "{}\n".format(score)
            res += "-------------------------------\n"
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_{}'.format(score)] == i)
                for candidate in candidates:
                    res += "Model with rank: {0}\n".format(i)
                    res += "Mean validation {0}: {1:.3f} (std: {2:.3f})\n".format(
                        score,
                        results['mean_test_{}'.format(score)][candidate],
                        results['std_test_{}'.format(score)][candidate])
                    res += "Parameters: {0}\n".format(results['params'][candidate])
                    res += "\n"

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(res)

            self._upload_to_bucket(filename, filename)

        logging.info(res)
