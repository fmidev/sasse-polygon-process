import numpy as np
import os, math, pyproj, yaml, logging, joblib, dask
from shapely.ops import transform
from functools import partial
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score, classification_report, make_scorer
from sklearn.preprocessing import label_binarize

from scipy.stats import expon
from scipy import interp

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from dask.distributed import Client
#from dask import delayed
#import dask.dataframe as dd
import dask, ast, itertools
import dask_ml.model_selection as dcv

from sklearn.metrics import roc_curve, auc, precision_score, f1_score, recall_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report

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


def cv(model, param_grid, X, y, n_iter=20): 
    """ Cross validation """

    cv_results = None
    print('..performing cv search...')
    searches = []

    # Define jobs
    random_search = dcv.RandomizedSearchCV(model, 
                                           param_grid, 
                                           n_iter=n_iter,
                                           cv=5,
                                           scoring=['f1_macro'], #, 'accuracy'],
                                           return_train_score=True,
                                           refit=False).fit(X, y)
    # Gather results
    cv_results = pd.DataFrame(random_search.cv_results_) #.head(1)    
    cv_results.sort_values(by=['mean_test_f1_macro'], inplace=True, ascending=False, ignore_index=True)
    print(cv_results.head())
    
    best_params = cv_results.loc[0,'params']
    model = model.set_params(**best_params)

    print('Using configuration: {}'.format(best_params))

    with joblib.parallel_backend('dask'):
        model.fit(X, y)
        
    return model, cv_results

def cv_(X, y, model, options, fh):
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


def gridcv(model, param_grid, X, y): 
    cv_results = None
    print('..performing cv search...')
    searches = []

    # Define jobs
    grid_search = dcv.GridSearchCV(model, 
                                   param_grid, 
                                   scoring=['f1_macro', 'f1_micro', 'accuracy'],
                                   return_train_score=True,
                                   refit=False,
                                   n_jobs=-1).fit(X, y)
    
    # Gather results
    cv_results = pd.DataFrame(grid_search.cv_results_) #.head(1)    
    cv_results.sort_values(by=['mean_test_f1_macro'], inplace=True, ascending=False, ignore_index=True)
    print(cv_results.head())
    
    best_params = cv_results.loc[0,'params']
    model = model.set_params(**best_params)

    print('Using configuration: {}'.format(best_params))

    with joblib.parallel_backend('dask'):
        model.fit(X, y)
        
    return model, cv_results

##############################################
# Visualisation functions
##############################################


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          cmap=plt.cm.YlOrBr,
                          filename=None,
                          fontsize=20):
    """
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    plt.rc('font', size=fontsize)

    fig, ax = plt.subplots(figsize=(6,6))
    np.set_printoptions(precision=2)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.grid(False, which='major')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    ax.xaxis.tick_top()
    plt.xticks(tick_marks, classes) #, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
    print(classification_report(y_true, y_pred))

def prec_rec_curve(y, y_pred, n_classes, fontsize=20):
    """
    Precision - Recall Curve
    """
    plt.rc('font', size=fontsize)
    colors=['xkcd:sky blue', 'xkcd:forest green', 'xkcd:dark red', 'xkcd:dark yellow']
    
    y = label_binarize(y, classes=np.arange(n_classes))

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(), y_pred.ravel())
    average_precision["micro"] = average_precision_score(y, y_pred, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    plt.figure(figsize=(12, 12))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y_ = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y_ >= 0], y_[y_ >= 0], color='gray', alpha=0.5)
        plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.9, y_[45] + 0.02))

    lines.append(l)
    labels.append('F1 curves')

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('Micro-average (area = {0:0.2f})'
    ''.format(average_precision["micro"]))

    for i in range(n_classes):
        l, = plt.plot(recall[i], precision[i], lw=2, color=colors[i])
        lines.append(l)
        labels.append('Class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(np.arange(.2, 1., .2))
    plt.xlabel('Recall', labelpad=20)
    plt.ylabel('Precision', labelpad=20)
    plt.title('Precision-Recall Curve', pad=20)
    plt.legend(lines, labels, loc=(0, -.2), ncol=2)

def feature_importance(data, feature_names = None, fontsize=20):
    """ Plot feature importance """

    fig, ax = plt.subplots(figsize=(24,18))

    plt.clf()
    plt.rc('font', size=fontsize)

    if feature_names is None:
        feature_names = range(0,len(data))
    else:
        plt.xticks(rotation=90, fontsize=fontsize)
        fig.subplots_adjust(bottom=0.5)

    plt.yticks(fontsize=fontsize*2/3)
    plt.bar(feature_names, data, align='center')
    plt.xlabel('Components', fontsize=fontsize, labelpad=20)
    plt.ylabel('Importance', fontsize=fontsize, labelpad=20)
    
    #ax.tick_params(axis='both', which='major', labelsize=fontsize)
    #ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    
    # plt.tight_layout()
    #fig.subplots_adjust(bottom=0.5)

    #self._save(plt, filename) 
    


def read_data(fname_train, fname_test, options):
    """ Read data from csv file """
    
    # Train
    data_train = pd.read_csv(fname_train)
    
    X_train = data_train.loc[:, options.feature_params]
    y_train = data_train.loc[:, options.label].values.ravel()
    
    print('Train data shape: {}'.format(X_train.shape))
    
    # Test
    if fname_test is not None:        
        data_test = pd.read_csv(fname_test)
                
        X_test = data_test.loc[:, options.feature_params]
        y_test = data_test.loc[:, options.label].values.ravel()

        print('Test data shape: {}'.format(X_test.shape))
    else:
        X_test, y_test = None, None
    
    return X_train, y_train, X_test, y_test

def plot_class_hist(data_train, data_test,title='', fontsize=10):

    fig, ax = plt.subplots(figsize=(15,4))
    plt.rc('font', size=fontsize)
    tickfontsize=0.8*fontsize
    
    ##### Plot 1
    ax = plt.subplot(1,2,1)

    data_train.loc[:, 'class'].hist(ax=ax, color='xkcd:tea')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax*1.1))
    
    plt.title('Train set', fontsize=fontsize)
    plt.ylabel('Record count', fontsize=fontsize)
    plt.xlabel('Class', fontsize=fontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.xticks(fontsize=tickfontsize)

    i=0
    for rect in ax.patches:
        if rect.get_height() > 0:
            height = rect.get_height()
            ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
        i+=1
    
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    ##### Plot 2 
    
    ax = plt.subplot(1,2,2)
    data_test.loc[:, 'class'].hist(ax=ax, color='xkcd:dust')    
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax*1.1))
    
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.title('Test set', fontsize=fontsize)
    plt.ylabel('Record count', fontsize=fontsize)
    plt.xlabel('Class', fontsize=fontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.xticks(fontsize=tickfontsize)
    
    i=0
    for rect in ax.patches:
        if rect.get_height() > 0:
            height = rect.get_height()
            ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
        i+=1
    
    plt.suptitle(title, x=.22, y=1.03)
    
def plot_roc(y, y_pred, n_classes=4, fontsize=20):
    """
    Plot multiclass ROC
    """
    
    colors=['xkcd:sky blue', 'xkcd:forest green', 'xkcd:dark red', 'xkcd:dark yellow']
    
    fig, ax1 = plt.subplots(figsize=(12,12))
    plt.clf()
    plt.rc('font', size=fontsize)
    
    y = label_binarize(y, classes=np.arange(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], threshhold = roc_curve(y[:, i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print('AUC for class {} is {}'.format(i, roc_auc[i]))

    # Compute average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(n_classes):        
        plt.plot(fpr[i], tpr[i], color=colors[i], label="Class {0} (AUC: {1:0.2f})".format(i, roc_auc[i]))

    plt.plot(fpr["macro"], tpr["macro"],
             label='Average (AUC: {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    plt.xlabel('False positive rate', fontsize=fontsize)
    plt.ylabel('True positive rate', fontsize=fontsize)
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    
    plt.yticks(fontsize=fontsize*2/3)
    plt.xticks(fontsize=fontsize*2/3)