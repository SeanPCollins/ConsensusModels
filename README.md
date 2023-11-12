# ConsensusModels
Code to calculate ideal consensus models based on results


The {file_name} should have no extension on it, however it needs to match with both a {file_name}.csv and {file_name}_FP.csv. The {file_name}.csv file contain both the information about the endpoints and model predictions that will be used to make the consensus models.  The {file_name}_FP.csv contains information regarding the fingerprints about the chemicals to use in the kNN approach.

Samples of both {file_name}.csv and {file_name}_FP.csv are providd, with the output file also given. For format, both {file_name}.csv and {file_name}_FP.csv need to have the same first column name. Multiple endpoints can be given in the {file_name}.csv, with columns of the observed results being given as '{endpoint} True' without the quatation marks. The columns following that are the predictions of the models for those substances. A second endpoint can be given in the same {file_name}.csv, if the observed values are once again put in a column name '{endpoint} True' without quatation marks.

The {file_name}_FP.csv needs to have matching indices to the {file_name}.csv, however the indices themselves can be arbitrarily chosen.

Columns indicating the SMILES are given in both inputs, however they are not nessecary. The {file_name}.csv only care about the first column (the index) and every column starting with the first column containing ' True' onawards. For the {file_name}_FP.csv, the required columns are the first column (the same index) and then the third column onwards. Currently the second column is ignored due to the program believing it to be a column containing SMILES information.

Once the input files are set, the program can be run, as stated above.

consensus_model.py {file_name}

This will try all avaialble possible combinations with the six current methods; majority, balanced accuracy weighted, modified score1, predictivity, modified predictivity, and k-Nearest Neighbors. The output will be a {file_name}_cconsensus_models.xlsx and a {file_name}.pkl. The xlsx file contains a worksheet for each endpoint, showing statistical results for all consensus models tested. The pickle (.pkl) file contains information about the consneus models which can be used to duplicate the results without needing to make the models again. To calculate predictions, the same input files as above need to be given, with slight modifications.

The {predict_name}_FP.csv is exactly the same as before, just with the same fingerprints as calculated before. For the {predict_name}.csv file, the predictions of the models, however there is no need for the ' True' columns for each endpoint. To make the predictions, simply run the command as follows.

consensus_model.py {predict_name} predict

For predictions, the program will look for all pkl files in the directory, and then run all available consensus models on the data.
