# MetaCellRank Computational Genomics Final Project

**Computational_Genomics_Final_Project_Submission.ipynb**

This file contains all the driver code that should be run. Input datasets are loaded from the scv package. (Sample output can be found in Example_Run.py.) Running all cells in the notebook takes up to 1 hour and 30 minutes. However, intermediate files named M_spliced.csv and M_unspliced.csv have been provided as pre-computed results for the most time-consuming part of the model. (These files were run with one set of parmeter configurations that are set as default in the notebook. If you want to change the parameters in the notebook, you will have to re-run from scratch.) If you are using the intermediate results that are provided in the csv files, uncomment the lines in the cell titled "Run Approach 1 Pipeline With Specified Parameters" and upload the two files into the files tab in the sidebar. More detailed instructions can be found in the notebook.

**requirements.txt**

This file shows all the required libraries needed to run MetaCellRank. DO NOT use this file to run the code, but only for inspection of the required libraries. The provided .ipynb file in Google Colab will make these imports and should be used to run the code. 

**Approach1.py**

This file contains all required imports, helper functions, dataset loading code, and driver to run approach 1. The dataset is loaded using the scVelo library, and does not require a separate upload. DO NOT use this file to run the code, but only for inspection. The provided .ipynb file in Google Colab should be used to run the code.

**Approach2.py**

This file contains all required imports, helper functions, dataset loading code, and driver to run approach 2. The dataset is loaded using the scVelo library, and does not require a separate upload. DO NOT use this file to run the code, but only for inspection. The provided .ipynb file in Google Colab should be used to run the code.

**Example_Run.ipynb**

This file contains an example Google Colab notebook of what the expected output should look like. DO NOT run this code. but can be used as reference.

**M_spliced.csv, M_unspliced.csv**

These files contain pre-computed results for the time-consuming step in the code. If these files are to be used, they must be uploaded into the Google Colab side bar and then the corresponding code block to load them in must be uncommented (explained in Google Colab in the cell titled "Run Approach 1 Pipeline With Specified Parameters"), and the current lines must be commented. 
