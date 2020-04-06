#!/bin/bash
folder=$1
files=""
for f in `aws s3 ls s3://fmi-asi-data-puusto/luke/2017/$folder/`;
do
  if [[ $f == *"tif" ]]; then
    files="$files /vsis3/fmi-asi-data-puusto/luke/2017/$folder/$f"
  fi
done

cmd="docker run -v ${HOME}:/root -v /tmp:/tmp --rm osgeo/gdal gdal_merge.py -init -init 32767 -a_nodata 32767 -o /tmp/${folder}_suomi_3067.tif ${files}"
echo $cmd && eval "$cmd"
cmd="docker run -v ${HOME}:/root -v /tmp:/tmp --rm osgeo/gdal gdalwarp -t_srs EPSG:4326 /tmp/${folder}_suomi_3067.tif /tmp/${folder}_suomi_4326.tif"
echo $cmd && eval "$cmd"
cmd="aws s3 cp /tmp/${folder}_suomi_4326.tif s3://fmi-asi-data-puusto/luke/2017/${folder}/${folder}_suomi_4326.tif"
echo $cmd && eval "$cmd"
