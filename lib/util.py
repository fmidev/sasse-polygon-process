import numpy as np
import os, math, pyproj, yaml
from shapely.ops import transform
from functools import partial

def get_savepath(options):
    """
    Get savepath based on given options
    """
    return options.model_savepath +'/'+options.model

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
