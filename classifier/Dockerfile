FROM python:3.7

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y libgdal-dev python-rtree
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install --upgrade numpy psycopg2-binary opencv-python-headless shapely geopandas pyyaml requests sqlalchemy boto3 rtree sklearn matplotlib

# Versions from Dask environment
RUN pip install --upgrade cloudpickle==1.2.2 blosc==1.7.0 dask==2.10.0 distributed==2.10.0 msgpack==0.6.1 tornado==6.0.3 fsspec>=0.3.3 scikit-learn==0.22.1

COPY . .

EXPOSE 5432
EXPOSE 80

CMD [ "python", "./" ]
