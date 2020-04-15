#!/bin/bash

# sudo mkdir -p /data && sudo chown 775 /data && sudo chcon -Rt svirt_sandbox_file_t /data
folder=$1
files=""
for f in `aws s3 ls s3://fmi-asi-data-puusto/luke/2017/$folder/`;
do
  if [[ $f == *"tif" ]]; then
    files="$files /vsis3/fmi-asi-data-puusto/luke/2017/$folder/$f"
  fi
done

BASE="/data"
FILE_3067="${BASE}/${folder}_suomi_3067.tif"
FILE_4326="${BASE}/${folder}_suomi_4326.tif"

cmd="docker run -v ${HOME}:/root -v /data:/data:rw --rm osgeo/gdal gdal_merge.py -init 32767 -a_nodata 32767 -o ${FILE_3067} ${files}"
echo $cmd 
$cmd
cmd="docker run -v ${HOME}:/root -v /data:/data:rw --rm osgeo/gdal gdalwarp -t_srs EPSG:4326 ${FILE_3067} ${FILE_4326}"
echo $cmd
$cmd

docker run -v $HOME:/root -v /data:/data:rw --rm osgeo/gdal gdalinfo -stat -hist $FILE_4326

cmd="aws s3 cp ${FILE_4326} s3://fmi-asi-data-puusto/luke/2017/${folder}/${folder}_suomi_4326.tif"
echo $cmd
$cmd
