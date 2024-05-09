# General imports
import argparse
from joblib import dump, load
import pandas as pd
import numpy as np
import os, sys

# Sklearn imports
from sklearn.linear_model import ElasticNet
from sklearn import set_config
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def readDataCSVs(x_csv):
    # Returns a dataframe with IDs dropped (TEMP)

    # Read in the data and make dataframes
    xdata_df = pd.read_csv(x_csv)

    # Drop the first columns of the dataframe (ID)
    xdata_filtered = xdata_df.drop(xdata_df.columns[[0]], axis=1)

    return xdata_filtered

def load_model(model_path):
# Loads the sklearn model and reutrns it so we can use it to run the data.

    # load the model from disk
    model = load(model_path)

    return model

def impute_missing_values(dataset):

    # Set up the imputer
    imputer = KNNImputer(n_neighbors=2, weights="uniform")

    # Create the imputed dataset
    x_imputed = imputer.fit_transform(dataset)

    return x_imputed

def log_transform(x):
    # Log transform whatever x is.
    log_transformed = np.log(x + 1)

    return log_transformed

def scale_dataset(x):

    # Transform data to between 0.001 and 1. We use 0.001 as a min so we can still do log transforms.
    scaler = MinMaxScaler(feature_range = (0.001, 1))
    scaler.fit(x)
    scaled_data = scaler.transform(x)
    
    # transform the entire dataset with log scaling.
    log_transfomed_x = scaled_data.transform(log_transform)

    return log_transfomed_x

def setTruncateOff():
    # removes truncation from pandas prints
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=sys.maxsize)

def writeResults(ydata):
    test_resultsfileName = str(args.savefile) + "_prediction.txt"
    test_resultsfile = open(test_resultsfileName, "w")

        # Write out the results line by line
    for point in ydata:
        test_resultsfile.write(str(point))
        test_resultsfile.write("\n")

    # Close the file
    test_resultsfile.close()

def checkSavePath(savelocation):
# Checks if a save path exists and if it doesen't creates the directory. Note: This expects linux formated "/" paths starting with ./ or ~/ as a format.

    # Split the save path up and determine the directory length.
    splitsave = savelocation.split("/")
    savepath = ""
    pathlength = len(splitsave)-1

    # Reconstruct the save path.
    for i in range(0,pathlength):
        if (i == 0):
            savepath = savepath + splitsave[i]
        else:
            savepath = savepath + "/" + splitsave[i]
    
    # Add the final slash
    savepath = savepath + "/"

    # Check to see if the directroy exists and if not create it.
    if os.path.exists(savepath):
        print ("Save path exists")
    else:
        print ("Save path does not exist. Creating directory " + savepath)
        try:
            os.makedirs(savepath)
        except:
            print("An exception occurred")

def polyTransValidationObject(yResults, p_fitted):
    # takes a results object and a fited polynomal and transforms all of the information in the results validation object via that polynomial

    # A place to stoore new values for prediction post transformation
    new_y_prediction = []

    # Rounding for final transform
    rounding = 15

    # Tranform the predictions based on the polynomial fit (3rd order recommended)
    for val in yResults:
        trans_val = round(p_fitted(float(val)),rounding)
        new_y_prediction.append(trans_val)

    return new_y_prediction

def final_output_transform(y_results):

    # A place to stoore new values for prediction post transformation
    new_y_prediction = []

    # Rounding for final transform
    rounding = 15

    # Tranform the predictions based on the polynomial fit (3rd order recommended)
    for val in y_results:
        transformed_value = ((val-60.7)/0.8159)+60.7
        trans_val = round(transformed_value, rounding)
        new_y_prediction.append(trans_val)

    return new_y_prediction

def main():

    # Set the output of our transformers to pandas.
    set_config(transform_output="pandas")

    # Turn of truncation settings for printing arrays and dataframes.
    setTruncateOff()

    # Start the run
    print("Running...")

    # Check the save path and if it doesn't exist make it
    checkSavePath(args.savefile)

    # Load the model from disk
    model = load_model(args.model)
    print("Running Model:")
    print("Alpha: " + str(model.alpha))
    print("L1: " + str(model.l1_ratio))

    # Read the CSV file of x inputs.
    xdata = readDataCSVs(args.xdata)

    # Impute missing x values
    print("Imputing missing values...")
    xdatai_df = impute_missing_values(xdata)

    # Scale the x data
    xscaled = scale_dataset(xdatai_df)

    # Predict the values
    yres = model.predict(xscaled)

    # Use model transformer if provided.
    if (args.transformer):
        transModel = load(str(args.transformer))
        yres = polyTransValidationObject(yres, transModel)

    # preform a final transform on the output
    yres = final_output_transform(yres)

    # Write model predictions to a results file.
    writeResults(yres)

    # Return nothing
    return 0

if __name__ == "__main__":

    # Parse paramaters that come in
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("xdata")
    parser.add_argument("savefile")
    parser.add_argument('-t', '--transformer', required=False)
    args = parser.parse_args()

    # Run the main fuction
    main()


    

