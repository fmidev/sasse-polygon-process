This repository is used to create training dataset and train a power outage model. Operative running code is in another repository. This code is open to support an related article. The data is, however, propriety data of the power distribution companies. The operative prediction code (running the model) is also in another, private, repository.

The overall process is following:
1. Fetch and prepare necessary data
2. Identify and track storm objects
3. Extract predictive features
4. Train classifier
5. Classify

## Fetch and prepare necessary data

Data consists of three parts: 1) ERA5 2) Luke forest inventory 3) power outage data.

### ERA5
ERA5 is fetched from https://cds.climate.copernicus.eu/ and is stored in AWS (private) bucket `fmi-era5-world-nwp-parameters`.

### Luke forest inventory
Luke forest inventory data is fetched from http://kartta.luke.fi/opendata/valinta.html and stored in (private) bucket `fmi-asi-data-puusto`. The data is stored as original (16m resolution) and lowres (1.6 km resoltion) GeoTiff and lowres GRIB files.

Following actions can be used to fetch new versions of the files:
1. Fetch the tiles from the service and upload them to the bucket with correct name (i.e. _luke/2017/fra_luokka/xx.tif_)  
2. Create composite GeoTiff: `bin/process_tree_files.sh fra_luokka)` (for all parameters)
3. Convert to GRIB: `bin/geotiff_to_grib` (process all parameters)

### Power outage data

Power outage data is received from power distribution companies.

## Track and extract predictive features

See https://github.com/fmidev/sasse-era5-smartmet-grid

## Classifier

## Dataset preparattion

Dataset is split to train and testset separately from training classifier. This ensures fair comparison and enable creating test examples. The split is conducted in notebook _dataset_split_.

### Train Classifier

Training the classifier is done partly with script `classifier/train_classifier.py` and partly in notebooks _train_and_validate_rfc_ (random search of the best hyperparameters), and _train_and_validate_mlp_.

The script reads relevant config as arguments and from `cnf/options.ini`. One can set config file and config section name as arguments `config_filename` and `config_name` respectively. Relevant setups at the moment are `thin` and `thin_energiateollisuus`.  Train and test data are always set as arguments `train_data` and `test_data`. If files do not exist locally, the script tries to fetch them from AWS bucket listed in `cnf/options.ini` variable `s3_bucket`. `model` (script supports svct, rfc, gnb, and gp). `dataset` argument is used to format model and results output path.

In practice, the script is ran with docker-compose. To train for example energiateollisuus dataset with 20 m/s threshold, one could use following commands:

1. `export model=rfc`
2. `export dataset=national_random_20`
3. `export train_data=data/energiateollisuus_random_20_thin_res.csv`
4. `export test_data=data/energiateollisuus_random_20_thin_test.csv`
5. `export config_name=thin_energiateollisuus`
6. `docker-compose run --rm cl`

### Classifying storm objects

For operational use, see https://github.com/fmidev/contour-storms

`classifier/create_examples.py` is used to create examples. Consult source code and docker-compose file for more details.
