FROM python:3.7

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y libgdal-dev python-rtree
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install --upgrade numpy psycopg2-binary opencv-python-headless shapely geopandas pyyaml requests sqlalchemy boto3 rtree sklearn matplotlib

# Versions from Dask environment
RUN pip install --upgrade dask_ml dask[dataframe]==2.12.0 distributed==2.12.0 lz4==3.0.2 blosc==1.8.3 cloudpickle==1.3.0 msgpack==1.0.0 numpy==1.18.1 toolz==0.10.0 tornado==6.0.4 imbalanced-learn

# pip install --upgrade pyproj pandas shapely geopandas dask_ml joblib scikit-learn==0.22.2.post1 dask[dataframe]==2.12.0 distributed==2.12.0 lz4==3.0.2 blosc==1.8.3 cloudpickle==1.3.0 msgpack==1.0.0 numpy==1.18.1 toolz==0.10.0 tornado==6.0.4 

COPY . .

EXPOSE 5432
EXPOSE 80

CMD [ "python", "./" ]
