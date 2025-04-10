# Ethiopian insurance claim prediction pipeline.

## Project outline:

-data: storage for raw data and retrieved batches;  
--raw: csv tables containing dataset;  
--batches: storage for retrieved raw batches

-models: storage for models and the best of them at the moment.  
--MLPRegressor: versions of MLPRegressor;  
--PARegressor: versions of PARegressor;  
--SGDRegressor: versions of SGDRegressor;  

-pipeline: logic for the whole pipeline  
--common.py: some constants;  
--data_analyser.py: DQ metrics, outliers cleansation, **EDA**  
--data_collector.py: batch retrieval, FS storage maintenance, meta collection, multiple source integration, logging  
--data_preprocessor.py: missing values, categorical and numerical features processing, feature engineering  
--model_training.py: model building, incremental learning, choice of three models by the best MSE. validation, model storage maintenance, ***visualisation of coefs***  
--model_manager.py: summary, inference  

-summary: files with validation metrics with each batch.

-config.json: pipeline config which is altered by the agent 

-vanilla: auxiliary directory for reset of the system  

-logging.conf: *configs for logger*

-main.py: point of entry & cmdline interface

-Makefile: auxiliary script for system reset & building

-meta.json: metadata of batches and overall data

# Usage

There are three command-line modes currently supported by our pipeline.

```bash
python3.11 main.py --mode "update"
python3.11 main.py --mode "inference" --path_to_data <path_to_unlabeled_csv>
python3.11 main.py --mode "summary"
```

Update mode must be ran at least once before inference and summary. 

To reset the system to its initial state, run 
```bash
make clean
```