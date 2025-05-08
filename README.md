# MetaCellRank Computational Genomics Final Project
Requirements.txt

This file shows all the required libraries needed to run MetaCellRank. DO NOT use this file to run the code, but only for inspection of the required libraries. The provided  .ipynb file in Google Colab will make these imports and should be used to run the code. 

Approach1.py

This file contains all required imports, helper functions, dataset loading code, and driver to run approach 1. The dataset is loaded using the scVelo library, and does not require a separate upload. DO NOT use this file to run the code, but only for inspection. The provided .ipynb file in Google Colab should be used to run the code.

Approach2.py

This file contains all required imports, helper functions, dataset loading code, and driver to run approach 2. The dataset is loaded using the scVelo library, and does not require a separate upload. DO NOT use this file to run the code, but only for inspection. The provided .ipynb file in Google Colab should be used to run the code.

Computational_Genomics_Final_Project_Submission.ipynb

This file contains all the code that should be run to test the code. The entire notebook, when run using the in-built methods, can take about 1 hour and 30 minutes to finish running. However, intermediate files called M_spliced.csv and M_unspliced.csv have been provided that have pre-computed results for the most time-consuming part of the model (however, these files were run with one set of parmeter configurations that are set as default in the notebook, if you want to change the parameters in the notebook, you will have to re-run from scratch). If you are using the intermediate results that are provided in the csv files, you will have to uncomment the lines in the cell titled "Run Approach 1 Pipeline With Specified Parameters" and also make sure to upload the two files into the files tab in the sidebar. The Google Colab notebook points out where to do this. 
