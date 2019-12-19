# loan-prediction-paid-on-time

In this project, we'll focus on the mindset of a conservative investor who only wants to invest in the loans that have a good chance of being paid off on time.

### Dataset Description
This dataset is from [Lending Club](https://www.lendingclub.com/info/download-data.action) where you can select a few different year ranges to download the datasets (in CSV format) for both approved and declined loans.
- `LCDataDictionary.xlsx`: contains information on the different column names
- `loans_2007.csv`: original dataset
- `filtered_loans_2007.csv`: dataset after removing some columns out (after running `NOTEBOOK - Data Cleaning.ipynb` file)
- `cleaned_loans_2007.csv`: dataset after imputing missing values and creating dummy variables (after running `NOTEBOOK - Data Cleaning.ipynb` file)

### Notebooks Description
1. `NOTEBOOK - Data Cleaning.ipynb`: 
  - This notebook deals with data cleaning and filtering columns that are not helpful. Also, it does some feature engineering, i.e. dummy variables
1. `NOTEBOOK - Predictive Modeling.ipynb`: 
  - This notebook does prediction, tries different models, tunes hyperparameters.
  
  
