import numpy as np
import os, math, pyproj, yaml, logging, joblib, dask
from shapely.ops import transform
from functools import partial
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score, classification_report, make_scorer
from scipy.stats import expon

def evaluate(model, options, data=None, dataset_file=None, fh=None, viz=None):
    """
    Evaluate dataset
    """

    if data is not None:
        X, y = data
    else:
        data = pd.read_csv(options.dataset_file)
        data = data.loc[data['weather_parameter'] == 'WindGust']

        X = data.loc[:, options.feature_params]
        y = data.loc[:, options.label].values.ravel()

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')

    logging.info('Accuracy: {}'.format(acc))
    logging.info('Precision: {}'.format(precision))
    logging.info('Recall: {}'.format(recall))
    logging.info('F1 score: {}'.format(f1))

    if fh is not None:
        error_data = {'acc': [acc],
                      'precision': [precision],
                      'recall': [recall],
                      'f1': [f1]}
        fname = '{}/test_validation_errors.csv'.format(options.output_path)
        fh.write_csv(error_data, filename=fname)

    # Confusion matrices
    fname = '{}/confusion_matrix_testset.png'.format(options.output_path)
    viz.plot_confusion_matrix(y, y_pred, np.arange(4), filename=fname)

    fname = '{}/confusion_matrix_testset_normalised.png'.format(options.output_path)
    viz.plot_confusion_matrix(y, y_pred, np.arange(4), True, filename=fname)

    # Precision-recall curve
    fname = '{}/precision-recall-curve_testset.png'.format(options.output_path)
    viz.prec_rec_curve(y, y_pred_proba, n_classes=4, filename=fname)

    # ROC
    fname = '{}/roc_testset.png'.format(options.output_path)
    viz.plot_roc(y, y_pred_proba, n_classes=4, filename=fname)

    # Feature importance
    if options.model == 'rfc':
        fname = '{}/feature_importance.png'.format(options.output_path)
        viz.rfc_feature_importance(model.feature_importances_, fname, feature_names=options.feature_params)

    logging.info('Validation report:\n{}'.format(classification_report(y_pred, y)))

def param_grid(model):
    """ Get params for KFold CV """
    if model == 'rfc':
        param_grid = {"n_estimators": [10, 100, 200, 800],
                      "max_depth": [3, 20, None],
                      "max_features": ["auto", "sqrt", "log2", None],
                      "min_samples_split": [2,5,10],
                      "min_samples_leaf": [1, 2, 4, 10],
                      "bootstrap": [True, False]}
    elif model == 'svc':
        param_grid = {"C": expon(scale=100),
                      "kernel": ['rbf', 'linear', 'sigmoid', 'poly'],
                      'degree': range(1,5)}
    else:
        raise "Not implemented"

    return param_grid

def cv(X, y, model, options, fh):
    """
    Cross-validate

    X : DataFrame | Array
        Features
    y : list
        labels
    model : obj
            scikit model
   options : obj
             options with at leas model, n_iter_search and output_path attributes
    fh : FileHandler
         file handler instance to report and store results

    return : model
    """


    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average = 'macro'),
               'recall': make_scorer(recall_score, average = 'macro'),
               'f1_macro': make_scorer(f1_score, average = 'macro'),
               'f1_weighted': make_scorer(f1_score, average = 'weighted')}


    random_search = RandomizedSearchCV(model,
                                       param_distributions=param_grid(options.model),
                                       n_iter=int(options.n_iter_search),
                                       scoring=scoring,
                                       cv=TimeSeriesSplit(),
                                       return_train_score=True,
                                       refit=False, # it's probably faster to retrain separately than keep probability True
                                       n_jobs=-1)

    logging.info('Starting 5-fold random search cross validation with {} iterations... X size is {}.'.format(options.n_iter_search, len(X)))

    #with joblib.parallel_backend('dask'):
    random_search.fit(X, y)

    logging.info("RandomizedSearchCV done.")

    t = scoring.keys()
    scores = []
    for s in t:
        scores.append('mean_test_{}'.format(s))

    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results.sort_values(by=['mean_test_f1_macro'], inplace=True, ascending=False, ignore_index=True)
    fname = options.output_path+'/random_search_cv_results.txt'
    fh.df_to_csv(cv_results, fname)
    logging.info("\n{}".format(cv_results.loc[:, scores]))

    # Fit with best params
    best_params = cv_results.loc[0,'params']
    if options.model in ['svc']:
        best_params['probability'] = True
    model.set_params(**best_params)

    #with joblib.parallel_backend('dask'):
    model.fit(X, y)

    #return random_search.best_estimator_
    return model

def feature_selection(X, y, model, options, fh):
    """
    Run feature selection process following:
    1. find feature importance by fitting RFC
    2. drop least important features one-by-one and run CV for new, supressed, dataset
    3. Store CV score of each step and draw graph
    """

    logging.info('Starting feature selection process...')

    logging.info('..traingin {} with {} samples'.format(options.model, len(X)))
    #with joblib.parallel_backend('dask'):
    model.fit(X, y)

    # Sort feature importances in descending order and rearrange feature names accordingly
    indices = np.argsort(model.feature_importances_)[::1]
    names = [options.feature_params[i] for i in indices]

    cv_results = None
    logging.info('..performing cv search...')
    #for i in range(0,len(names)-4):
    for i in range(0,5):
        logging.info('...with {} parameters'.format(len(names)-i))
        data = X.loc[:,names[i:]]
        random_search = RandomizedSearchCV(model,
                                           param_distributions=param_grid(options.model),
                                           n_iter=int(options.n_iter_search),
                                           scoring=['f1_macro', 'f1_micro', 'accuracy'],
                                           return_train_score=True,
                                           refit=False,
                                           n_jobs=-1)

        #try:
        #    with joblib.parallel_backend('dask'):
        random_search.fit(data, y)
#        except AttributeError:
#            logging.warning('AttributeError while fitting. Trying again.')
#            with joblib.parallel_backend('dask'):
#                random_search.fit(data, y)

        if cv_results is None:
            cv_results = pd.DataFrame(random_search.cv_results_) #.head(1)
            cv_results['Number of parameters'] = len(names)-i
        else:
            res_df = pd.DataFrame(random_search.cv_results_) #.head(1)
            res_df['Number of parameters'] = len(names)-i
            cv_results = pd.concat([cv_results, res_df], ignore_index=True)

        #cv_results.append(dask.delayed(train)(random_search, data, y))

    #cv_results = dask.compute(*cv_results)
    logging.info('..cv search done')
    print(cv_results)
    cv_results.sort_values(by=['mean_test_f1_macro'], inplace=True)
    print(cv_results)

    # Save and visualise results
    fname = '{}/feature_selection_results.csv'.format(options.output_path)
    fh.df_to_csv(cv_results, fname, fname)

    logging.info('..refitting with best model params')
    model.set_params(**cv_results.loc[0,'params'])
    params = names[cv_results.loc[0, 'Number of parameters']:]
    data = X.loc[:, params]

    #with joblib.parallel_backend('dask'):
    model.fit(data, y)

    return model, params


def get_param_names(config_filename, shortnames=True):
    """ Get param names, partly from config and partly as hard coded """

    with open(config_filename) as f:
        file_content = f.read()

        config_dict = yaml.load(file_content, Loader=yaml.FullLoader)

        params = config_dict['params']
        met_params = set()
        for param, info in params.items():
            for f in info['aggregation']:
                if shortnames:
                    met_params.add(f[1:]+' '+info['name'])
                else:
                    met_params.add(f+'{'+param+'}')

    met_params = list(met_params)
    polygon_params = ['speed_self', 'angle_self', 'area_m2', 'area_diff', 'low_limit']
    features = polygon_params + met_params

    meta_params = ['id', 'storm_id', 'point_in_time', 'weather_parameter', 'high_limit', 'transformers', 'all_customers', 'outages', 'customers']
    labels = ['class', 'class_customers']

    all_params = features + meta_params + labels

    return features, meta_params, labels, all_params


def surrounding_indexes(pixel_coord, window, boundary):
    """ Returns the indexes of the pixels surrounding the given
    pixel coordinate.
    """
def circular_kernel(radius, fill_value):
    """Returns a rectangular numpy array of shape (2*radius+1, 2*radius+1)
    where all values within radius from the center are set to fill_value
    """
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = fill_value
    return kernel

def minimum_nonzero_bbox(base_array):
    """Returns a minimum bounding box for non zero values
    """
    tmp_array = np.argwhere(base_array)
    (ystart, xstart), (ystop, xstop) = tmp_array.min(0), tmp_array.max(0)
    return (ystart, ystop, xstart, xstop)

def rectangle_intersection(r1, r2):
    """Returns the intersection of two rectangular areas
    in form (ul_y, ul_x, w, h). Retruns None if rectangles
    do not intersect.
    """
    if r1[0] > r2[0]:
        y = r1[0]
    else:
        y = r2[0]
    if r1[1] > r2[1]:
        x = r1[1]
    else :
        x = r2[1]

    w = min((r1[1] + r1[2], r2[1] + r2[2])) - x
    h = min((r1[0] + r1[3], r2[0] + r2[3])) - y

    if w <= 0 or h <= 0:
        return None
    else:
        return (y, x, w, h)


def insert_points(base_array, points, accumulate=False):
    if accumulate:
        for y, x in iter(points):
            try:
                base_array[y, x] += 1
            except IndexError:
                pass
    else:
        for y, x in iter(points):
            try:
                base_array[y,x] = 1
            except IndexError:
                pass

def insert_array(base_array, window, y, x, accumulate=False, insertmax=False):
    """
    function inserts the values of window array to base_array so that
    the upper left corner of the window is at point y, x. If window
    positioned at y, x doesn't intersect base_array, base_array stays
    unchanged.

    Parameters
    ----------
    base_array : np.ndarray 2d
        Array where new values area inserted
    window : np.nd_array
        Array that is inserted to base_array
    y : int
        insertion row coordinate
    x : int
        insertion column coordinate
    accumulate : bool
        If accumulate is set to True window values are accumulated on base_array values
        otherwise widow values overwrite the base_array values.
    insertmax : bool
        If base array contains values where window should be inserted, choose the max values
        at each position to be inserted on base_array
    """
    h1, w1 = base_array.shape
    h2, w2 = window.shape

    # x and y are within base array
    if 0 <= y < h1:
        y_min1 = y
        y_min2 = 0

        if y + h2 > h1:
            y_max1 = h1
            y_max2 = h1 - y
        else:
            y_max1 = y + h2
            y_max2 = h2
    elif -h2 < y < 0:
        y_min1 = 0
        y_max1 = y + h2
        y_min2 = -y
        y_max2 = h2
    if 0 <= x < w1:
        x_min1 = x
        x_min2 = 0
        if x + w2 > w1:
            x_max1 = w1
            x_max2 = h1 - x
        else:
            x_max1 = x + w2
            x_max2 = w2
    elif -w2 < x < 0:
        x_min1 = 0
        x_max1 = x + w2
        x_min2 = -x
        x_max2 = w2
    try:
        if accumulate:
            base_array[y_min1:y_max1, x_min1:x_max1] += window[y_min2:y_max2, x_min2:x_max2]
        elif insertmax:
            # if base_array contains values at the area of window, select the maximum values from window and base_array
            max_window = np.amax([base_array[y_min1:y_max1, x_min1:x_max1], window[y_min2:y_max2, x_min2:x_max2]], axis=0)
            base_array[y_min1:y_max1, x_min1:x_max1] = max_window
        else:
            base_array[y_min1:y_max1, x_min1:x_max1] = window[y_min2:y_max2, x_min2:x_max2]
    except:
        pass



def insert_array2(base_array, window, samples, accumulate=False):
    """
    function inserts the values of window array to base_array so that
    the upper left corner of the window is at point y, x. If window
    positioned at y, x doesn't intersect base_array, base_array stays
    unchanged.

    Parameters
    ----------
    base_array : np.ndarray 2d
        Array where new values area inserted
    window : np.nd_array
        Array that is inserted to base_array
    y : int
        insertion row coordinate
    x : int
        insertion column coordinate
    accumulate : bool
        If accumulate is set to True window values are accumulated on base_array values
        otherwise widow values overwrite the base_array values.
    """
    for y, x in samples:
        h1, w1 = base_array.shape
        h2, w2 = window.shape

        # x and y are within base array
        if 0 <= y < h1:
            y_min1 = y
            y_min2 = 0

            if y + h2 > h1:
                y_max1 = h1
                y_max2 = h1 - y
            else:
                y_max1 = y + h2
                y_max2 = h2
        elif -h2 < y < 0:
            y_min1 = 0
            y_max1 = y + h2
            y_min2 = -y
            y_max2 = h2
        if 0 <= x < w1:
            x_min1 = x
            x_min2 = 0
            if x + w2 > w1:
                x_max1 = w1
                x_max2 = h1 - x
            else:
                x_max1 = x + w2
                x_max2 = w2
        elif -w2 < x < 0:
            x_min1 = 0
            x_max1 = x + w2
            x_min2 = -x
            x_max2 = w2
        try:
            if accumulate:
                base_array[y_min1:y_max1, x_min1:x_max1] += window[y_min2:y_max2, x_min2:x_max2]
            else:
                base_array[y_min1:y_max1, x_min1:x_max1] = window[y_min2:y_max2, x_min2:x_max2]
        except:
            pass


def bearing(pt1, pt2):
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    angle = math.degrees(math.atan2(y_diff, x_diff))
    if angle < 0: angle += 360
    return angle

def speed(row1, row2, threshold = None, missing = None):
    c1 = transform(
    partial(
        pyproj.transform,
        pyproj.Proj(init='EPSG:4326'),
        pyproj.Proj(
            proj='aea',
            lat_1=row1.geom.bounds[1],
            lat_2=row1.geom.bounds[3])),
        row1.geom).centroid

    c2 = transform(
    partial(
        pyproj.transform,
        pyproj.Proj(init='EPSG:4326'),
        pyproj.Proj(
            proj='aea',
            lat_1=row2.geom.bounds[1],
            lat_2=row2.geom.bounds[3])),
        row2.geom).centroid

    dist = c1.distance(c2) / 1000
    if threshold is not None and dist > threshold:
        dist = missing

    return dist
