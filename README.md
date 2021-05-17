# covid19_pred
Stratified-bagging models for the prediction of SARS-CoV-2 Inhibitors

The jupyter notebook for the prediction of SARS-CoV-2 Inhibitors
 using Stratified Bagging models and Morgan fingerprints.

Installation and dependencies
Anaconda (https://docs.anaconda.com/anaconda/navigator/install/) >= 1.9.12; 
Python >= 3.7.7; 
Pandas >= 1.0.3; 
numpy >= 1.18.1

Usage

1)	From the anaconda navigator, start the jupyter notebook and load the file ‘COVID_Prediction_StratifiedBagging_MorganFP.ipynb’ available under the folder ‘CPE_StratifiedBagging’
2)	Calculate the Morgan fingerprint for our test compounds.
3)	Place the generated file with the calculated descriptor in the folder ‘test_data’. A sample file is available (sample_test_file.csv).
4)	In the jupyter notebook ‘COVID_Prediction_StratifiedBagging_MorganFP.ipynb’, change the name of the test file (3rd cell; #Path to the test file)
5)	Execute the notebook.
6)	The SARS-CoV-2 Inhibitor’s prediction using Stratified Bagging models and Morgan fingerprints will be saved in the ‘output_predictions’ folder.

The folder ‘SB_models’ also contains Stratified Bagging models generated for Avalon Fingerprint and rdKit descriptors (physicochemical properties).
