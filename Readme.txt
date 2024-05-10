Description:
This code is a public implementation of the main cerebrospinal fluid (CSF) proteomics age prediction models found in "An Interpretable Machine Learning-Based Cerebrospinal Fluid Proteomics Clock for Predicting Age Reveals Novel Insights into Brain Aging." written by Justin Melendez et al. and published in Aging Cell. For assistance please contact justin.melendez@wustl.edu in the Bateman Lab at Washington University in Saint Louis.

Requirements:

Python 3 is required to run this software as well as the following dependencies. Please install the following python packages before running models.

argparse
joblib
pandas
numpy
scikit-learn

Instructions:
The models implemented here take CSF proteomic data from the SomaScan 7K Assay by SomaLogic and predict the chronological age of individuals. There are two main models described in the paper. A "Full Model" which uses 1,157 protein measurements to make the prediction and a "Minimal Model" which uses just 109 proteins. Please refer to the paper and the instructions below before attempting to run any models. It is very important the input data be structured properly for each model or the code will not run properly. For this reason, template files have been provided with the order and names of the SomaLogic features to be used. Please refer to the Extended Materials excel file from the paper for detailed information about each protein feature (including full protein names and gene ids).

To run the full model with your data, run the following command replacing the 3rd argument with the location of your proteomics data and the 4th argument with the location and name of your desired results file. Note that your proteomics data must be in the exact form and order as the "full_template.csv" template provided in the "Templates" folder for the program to run properly.

python3 CSF_Proteomics_Aging_Clock.py ./Models/Full/full.model -t ./Models/Full/full.transformer ./YourProteomics.csv ./Results/YourResult


To run the minimal model with your data, run the following command replacing the 3rd argument with the location of your proteomics data and the 4th argument with the location and name of your desired results file. Note that your proteomics data must be in the exact form and order as the "minimal_template.csv" template provided in the "Templates" folder for the program to run properly and that this is a different set of proteins than the full model.

python3 CSF_Proteomics_Aging_Clock.py ./Models/Minimal/minimal_109.model -t ./Models/Minimal/minimal_109.transformer ./YourProteomics.csv ./Results/YourResult
