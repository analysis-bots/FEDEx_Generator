# FEDEx_Generator test module
This module contains tests and scripts for running automatic tests on the FEDEx library.\
\
The tests work by comparing a "screenshot" of results and computations from a time 
we knew the library was working correctly, with results and computations done by the current code.
## File structure
The module is structured as follows:
```
tests/
│ resources/
│ │ datasets/
│ │ │ csv files
│ │ queries_files/
│ │ │ txt files
│ │ results_files/
│ │ │ json files
│ │ default_datasets.json
│ Python files, readme, etc.
```
The resources folder contains the datasets, queries and results files used for testing.\
The default_datasets.json file contains information on the names of our default testing datasets, as well as how to download them if needed.

### Dataset files
The datasets are stored in csv format, and the tests currently only support csv files.\
The tests know to look in this folder for the datasets in case a user specified path is not given or is incorrect.

### Queries files
The queries are stored in txt format, and should be in the following format:
```
# Comment lines start with '#'
# Query format: (column_name, operator, explainer, dict of query parameters)
# Example queries:

(label, ==, fedex, {value: <=50K, top_k: 10})
(workclass, groupby, fedex, {agg_function: mean, top_k: 10, select_columns: [age ; fnlwgt ; education-num ; capital-gain ; capital-loss ; hours-per-week]})
(age, join, fedex, {})
(workclass, groupby, outlier, {agg_function: mean, top_k: 10, select_columns: age, dir: high, target: Never-worked})

# You can also specify a global select, to choose only a subset of columns to use in all columns of the dataset.
# For example:

GlobalSelect=[MSSubClass, LotArea, OverallQual, OverallCond, YearBuilt, 1stFlrSF, 2ndFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, GarageCars, PoolArea, YrSold, SalePrice]

# If you want to specify a global select on the second dataset, use GlobalSelectSecond.
# Please note that everything is case sensitive.
```
These queries are used with the `extract_intermediate_results.py` script to generate the result files.\
There should be 4 of these files provided - one for each dataset.\
You can add more queries or additional query files, if you want to test more queries.

### Results files
The result files are json files that contain all the intermediate results of the queries.\
These are the files required for running the tests.\
Unless specified otherwise, the tests will look in this folder and load all the results files for the datasets.

## Running the tests
To run the tests with the files in the results_files directory, simply run the test_query_explainers.py script:
```bash
python test_query_explainers.py
```
This will run tests according to the queries and datasets in each result file.\
Generally speaking, the 4 provided result files should be enough to test the library.\
If you want to specify a result file or files, add command line arguments:
```bash
python test_query_explainers.py <path_to_result_file1> <path_to_result_file2> ...
```
Those result files should always be files extracted by the `extract_intermediate_results.py` script.\
\
The script will run the queries in the result files on the datasets using the current code, compare the results to those saved in
the result files, and give a comprehensive report on the tests.\

## Creating new result files
To create new result files, you can use the `extract_intermediate_results.py` script.\
This script accepts 4 arguments, as follows:
```bash
python extract_intermediate_results.py <path_to_dataset> <path_to_query_file> <path_to_output_file> <path_to_second_dataset: optional>
```
The second dataset is required only if you are using a join query, and it can be the same dataset as the first one.\
\
The script will run the queries in the query file on the dataset, and extract the intermediate results of the queries, 
creating a results file that can be used for testing.\

## Downloading the default datasets
To download all of the default datasets, you can run the `download_datasets_util.py` script:
```bash
python download_datasets_util.py
```
This script will automatically create the `resources/datasets` folder and download the datasets into it.\
The default datasets are those present in the pd-explain repository:
- adult.csv
- spotify_all.csv
- bank_churners_user_study.csv
- houses.csv
