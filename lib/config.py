import os
from configparser import ConfigParser

def read_options(options):

    def _path(name, root_dir):
        ''' Read path from options and create it if not exists'''
        val = getattr(options, name, None)
        if val is None or val == 'None':
            val = root_dir+'/'+options.model+'/'+options.dataset+'/'+options.config_name

        if not os.path.exists(val):
            os.makedirs(val)

        setattr(options, name, val)

    def _fval(name, default=None):
        ''' Convert float val to float taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = float(val)
        else:
            val = default
        setattr(options, name, val)

    def _bval(name, default=False):
        ''' Convert option from int to bool'''
        val = getattr(options, name, False)
        if int(val) == 1: val = True
        else: val = default
        setattr(options, name, val)

    def _intval(name, default=None):
        ''' Convert int val to integer taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = int(val)
        else:
            val = default
        setattr(options, name, val)


    parser = ConfigParser()
    parser.read(options.config_filename)

    if parser.has_section(options.config_name):
        params = parser.items(options.config_name)
        for param in params:
            if getattr(options, param[0], None) is None:
                setattr(options, param[0], param[1])

        options.feature_params = options.feature_params.split(',')
        options.label = options.label.split(',')
        options.meta_params = options.meta_params.split(',')

        options.dataset_file = 'data/classification_dataset_{}.csv'.format(options.dataset)

        if hasattr(options, 'test_dataset'):
            options.test_dataset_file = 'data/classification_dataset_{}.csv'.format(options.test_dataset)

        _path('save_path', 'models')
        options.save_file = options.save_path+'/model.pkl'
        _path('output_path', 'results')

        # common / several
        _bval('cv')
        _bval('feature_selection')
        _bval('pca')
        _bval('whiten')
        _bval('normalize')
        _bval('balance')
        _bval('smote')
        _bval('evaluate')
        _bval('save_data')
        _intval('n_iter_search', 10)
        _bval('debug')

        # linear regression
        _fval('alpha')
        _fval('eta0')
        _fval('power_t')
        _bval('shuffle')

        # RFC
        _bval('bootstrap')
        _intval('n_estimators')
        _intval('min_samples_split')
        _intval('min_samples_leaf')
        _intval('max_depth')

        # SVC
        _bval('probability')
        _fval('penalty')
        _fval('gamma')

        # other
        _intval('pca_components')

        return options
    else:
        raise Exception('Section {} not found in the {} file'.format(options.config_name, options.config_filename))

    return tables
