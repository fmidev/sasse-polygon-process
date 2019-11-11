FROM python:3.7

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y libgdal-dev
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install --upgrade numpy psycopg2-binary opencv-python-headless shapely geopandas
#RUN pip install --upgrade yaml

COPY . .

EXPOSE 5432

CMD [ "python", "./" ]
