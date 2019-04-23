
In this project, we'll focus on the mindset of a conservative investor who only wants to invest in the loans that have a good chance of being paid off on time.

### Define the problem statement for this machine learning project:

- Can we build a machine learning model that can accurately predict if a borrower will pay off their loan on time or not?


```python
# import pandas as pd

# # removing the first line because it contains extraneous text
# loans_2007 = pd.read_csv("LoanStats3a.csv", skiprows=1)

# # removing the `desc` (long text explanation), and `url`(only accessed by investor)
# loans_2007 = loans_2007.drop(['desc', 'url'], axis=1)

# # removing all columns containing more than 50% missing values
# half_count = len(loans_2007) / 2

# loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)

# loans_2007.to_csv('loans_2007_myone.csv', index=False)
```


```python
loans_2007 = pd.read_csv("loans_2007.csv")
loans_2007.head()
```

    c:\python36\lib\site-packages\IPython\core\interactiveshell.py:2785: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>last_pymnt_amnt</th>
      <th>last_credit_pull_d</th>
      <th>collections_12_mths_ex_med</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>171.62</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>119.66</td>
      <td>Sep-2013</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>649.91</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>357.48</td>
      <td>Apr-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>1311748.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>...</td>
      <td>67.79</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>




```python
loans_2007.shape
```




    (42538, 52)




```python
loans_2007.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 42538 entries, 0 to 42537
    Data columns (total 52 columns):
    id                            42538 non-null object
    member_id                     42535 non-null float64
    loan_amnt                     42535 non-null float64
    funded_amnt                   42535 non-null float64
    funded_amnt_inv               42535 non-null float64
    term                          42535 non-null object
    int_rate                      42535 non-null object
    installment                   42535 non-null float64
    grade                         42535 non-null object
    sub_grade                     42535 non-null object
    emp_title                     39909 non-null object
    emp_length                    41423 non-null object
    home_ownership                42535 non-null object
    annual_inc                    42531 non-null float64
    verification_status           42535 non-null object
    issue_d                       42535 non-null object
    loan_status                   42535 non-null object
    pymnt_plan                    42535 non-null object
    purpose                       42535 non-null object
    title                         42522 non-null object
    zip_code                      42535 non-null object
    addr_state                    42535 non-null object
    dti                           42535 non-null float64
    delinq_2yrs                   42506 non-null float64
    earliest_cr_line              42506 non-null object
    inq_last_6mths                42506 non-null float64
    open_acc                      42506 non-null float64
    pub_rec                       42506 non-null float64
    revol_bal                     42535 non-null float64
    revol_util                    42445 non-null object
    total_acc                     42506 non-null float64
    initial_list_status           42535 non-null object
    out_prncp                     42535 non-null float64
    out_prncp_inv                 42535 non-null float64
    total_pymnt                   42535 non-null float64
    total_pymnt_inv               42535 non-null float64
    total_rec_prncp               42535 non-null float64
    total_rec_int                 42535 non-null float64
    total_rec_late_fee            42535 non-null float64
    recoveries                    42535 non-null float64
    collection_recovery_fee       42535 non-null float64
    last_pymnt_d                  42452 non-null object
    last_pymnt_amnt               42535 non-null float64
    last_credit_pull_d            42531 non-null object
    collections_12_mths_ex_med    42390 non-null float64
    policy_code                   42535 non-null float64
    application_type              42535 non-null object
    acc_now_delinq                42506 non-null float64
    chargeoff_within_12_mths      42390 non-null float64
    delinq_amnt                   42506 non-null float64
    pub_rec_bankruptcies          41170 non-null float64
    tax_liens                     42430 non-null float64
    dtypes: float64(30), object(22)
    memory usage: 16.9+ MB
    

# Let's break up the columns into 3 groups of 18 columns

We will pay attention to any features that:
- leak information from the future (after the loan has already been funded)
- don't affect a borrower's ability to pay back a loan (e.g. a randomly generated ID value by Lending Club)
- formatted poorly and need to be cleaned up
- require more data or a lot of processing to turn into a useful feature
- contain redundant information


### First group of features
After analyzing each column in the first group, we can conclude that the following features need to be removed:
- `id`
- `member_id`
- `funded_amnt`
- `funded_amnt_inv`
- `grade`
- `sub_grade`
- `emp_title`
- `issue_d`


```python
# id and member_id are already removed previously due to 50% of missing values
loans_2007  = loans_2007.drop(["id","member_id","funded_amnt","funded_amnt_inv","grade","sub_grade","emp_title","issue_d"], axis=1)
```

### Second group of features
 The group of features to be removed
- `zip_code`
- `out_prncp`
- `out_prncp_inv`
- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_prncp`


```python
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
```

### Thrid group of features

In the last group of columns, we need to drop the following columns:
- `total_rec_int`
- `total_rec_late_fee`
- `recoveries`
- `collection_recovery_fee`
- `last_pymnt_d`
- `last_pymnt_amnt`

All of these columns leak data from the future, meaning that they're describing aspects of the loan after it's already been fully funded and started to be paid off by the borrower.


```python
loans_2007 = loans_2007.drop(["total_rec_int",
"total_rec_late_fee",
"recoveries",
"collection_recovery_fee",
"last_pymnt_d",
"last_pymnt_amnt"], axis=1)

print(loans_2007.iloc[0])
print(loans_2007.shape[1])
```

    loan_amnt                            5000
    term                            36 months
    int_rate                           10.65%
    installment                        162.87
    emp_length                      10+ years
    home_ownership                       RENT
    annual_inc                          24000
    verification_status              Verified
    loan_status                    Fully Paid
    pymnt_plan                              n
    purpose                       credit_card
    title                            Computer
    addr_state                             AZ
    dti                                 27.65
    delinq_2yrs                             0
    earliest_cr_line                 Jan-1985
    inq_last_6mths                          1
    open_acc                                3
    pub_rec                                 0
    revol_bal                           13648
    revol_util                          83.7%
    total_acc                               9
    initial_list_status                     f
    last_credit_pull_d               Jun-2016
    collections_12_mths_ex_med              0
    policy_code                             1
    application_type               INDIVIDUAL
    acc_now_delinq                          0
    chargeoff_within_12_mths                0
    delinq_amnt                             0
    pub_rec_bankruptcies                    0
    tax_liens                               0
    Name: 0, dtype: object
    32
    

# Target Column

We use the `loan_status` column, since it's the only column that directly describes if a loan was paid off on time, had delayed payments, or was defaulted on the borrower. 


```python
loans_2007['loan_status'].value_counts()
```




    Fully Paid                                             33136
    Charged Off                                             5634
    Does not meet the credit policy. Status:Fully Paid      1988
    Current                                                  961
    Does not meet the credit policy. Status:Charged Off      761
    Late (31-120 days)                                        24
    In Grace Period                                           20
    Late (16-30 days)                                          8
    Default                                                    3
    Name: loan_status, dtype: int64



Since we're interested in predicting which of these 2 values a loan will fall under and only the `Fully Paid` and `Charged Off` values describe the final outcome of the loan, we can treat the problem as a <b>binary classification</b>


```python
loans_2007 = loans_2007[(loans_2007["loan_status"]=="Fully Paid") | (loans_2007["loan_status"]=="Charged Off")]

status_replace = {
    "loan_status":{
        "Fully Paid": 1,
        "Charged Off": 0 
    }
}

loans_2007 = loans_2007.replace(status_replace)
```


```python
loans_2007.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>...</th>
      <th>initial_list_status</th>
      <th>last_credit_pull_d</th>
      <th>collections_12_mths_ex_med</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>Verified</td>
      <td>1</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>0</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Sep-2013</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>1</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>Source Verified</td>
      <td>1</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Apr-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5000.0</td>
      <td>36 months</td>
      <td>7.90%</td>
      <td>156.46</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>36000.0</td>
      <td>Source Verified</td>
      <td>1</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Jan-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



### Removing single value columns

Let's look for any columns that contain only one unique value and remove them. These columns won't be useful for the model since they don't add any information to each loan application

First, drop the null values then compute the number of unique values:


```python
drop_columns = list()

for col in loans_2007.columns:
    col_series = loans_2007[col].dropna().unique()
    
    if len(col_series) == 1:
        drop_columns.append(col)
        
loans_2007 = loans_2007.drop(drop_columns, axis=1)
print(drop_columns)
```

    ['pymnt_plan', 'initial_list_status', 'collections_12_mths_ex_med', 'policy_code', 'application_type', 'acc_now_delinq', 'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens']
    


```python
loans_2007.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>loan_status</th>
      <th>purpose</th>
      <th>...</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>last_credit_pull_d</th>
      <th>pub_rec_bankruptcies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>Verified</td>
      <td>1</td>
      <td>credit_card</td>
      <td>...</td>
      <td>0.0</td>
      <td>Jan-1985</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>13648.0</td>
      <td>83.7%</td>
      <td>9.0</td>
      <td>Jun-2016</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>0</td>
      <td>car</td>
      <td>...</td>
      <td>0.0</td>
      <td>Apr-1999</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1687.0</td>
      <td>9.4%</td>
      <td>4.0</td>
      <td>Sep-2013</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>1</td>
      <td>small_business</td>
      <td>...</td>
      <td>0.0</td>
      <td>Nov-2001</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2956.0</td>
      <td>98.5%</td>
      <td>10.0</td>
      <td>Jun-2016</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>Source Verified</td>
      <td>1</td>
      <td>other</td>
      <td>...</td>
      <td>0.0</td>
      <td>Feb-1996</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>5598.0</td>
      <td>21%</td>
      <td>37.0</td>
      <td>Apr-2016</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5000.0</td>
      <td>36 months</td>
      <td>7.90%</td>
      <td>156.46</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>36000.0</td>
      <td>Source Verified</td>
      <td>1</td>
      <td>wedding</td>
      <td>...</td>
      <td>0.0</td>
      <td>Nov-2004</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>7963.0</td>
      <td>28.3%</td>
      <td>12.0</td>
      <td>Jan-2016</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
loans_2007.to_csv("filtered_loans_2007.csv", index=False)
```


```python
loans = pd.read_csv("filtered_loans_2007.csv")
null_counts = loans.isnull().sum()
print(null_counts)
```

    loan_amnt                  0
    term                       0
    int_rate                   0
    installment                0
    emp_length              1036
    home_ownership             0
    annual_inc                 0
    verification_status        0
    loan_status                0
    purpose                    0
    title                     11
    addr_state                 0
    dti                        0
    delinq_2yrs                0
    earliest_cr_line           0
    inq_last_6mths             0
    open_acc                   0
    pub_rec                    0
    revol_bal                  0
    revol_util                50
    total_acc                  0
    last_credit_pull_d         2
    pub_rec_bankruptcies     697
    dtype: int64
    

## Handling Missing Values


```python
loans = loans.drop(["pub_rec_bankruptcies"], axis=1)

loans = loans.dropna(axis=0)

print(loans.dtypes.value_counts())
```

    object     11
    float64    10
    int64       1
    dtype: int64
    

## Converting Text Columns to Numerical data types


```python
object_columns_df = loans.select_dtypes(include=["object"])
object_columns_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>term</th>
      <th>int_rate</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>verification_status</th>
      <th>purpose</th>
      <th>title</th>
      <th>addr_state</th>
      <th>earliest_cr_line</th>
      <th>revol_util</th>
      <th>last_credit_pull_d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36 months</td>
      <td>10.65%</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>credit_card</td>
      <td>Computer</td>
      <td>AZ</td>
      <td>Jan-1985</td>
      <td>83.7%</td>
      <td>Jun-2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60 months</td>
      <td>15.27%</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>car</td>
      <td>bike</td>
      <td>GA</td>
      <td>Apr-1999</td>
      <td>9.4%</td>
      <td>Sep-2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36 months</td>
      <td>15.96%</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>small_business</td>
      <td>real estate business</td>
      <td>IL</td>
      <td>Nov-2001</td>
      <td>98.5%</td>
      <td>Jun-2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36 months</td>
      <td>13.49%</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>other</td>
      <td>personel</td>
      <td>CA</td>
      <td>Feb-1996</td>
      <td>21%</td>
      <td>Apr-2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36 months</td>
      <td>7.90%</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>wedding</td>
      <td>My wedding loan I promise to pay back</td>
      <td>AZ</td>
      <td>Nov-2004</td>
      <td>28.3%</td>
      <td>Jan-2016</td>
    </tr>
  </tbody>
</table>
</div>



Some of the columns seem like they represent categorical values:
- `home_ownership`
- `verification_status`
- `emp_length`
- `term`
- `addr_state`
- `purpose`
- `title`

For `purpose` and `title`, it seems like these columns could reflect the same information.

There are also some columns that represent numeric values, that need to be converted:
- `int_rate`
- `revol_util`

Lastly, some of the columns contain date values that would require a good amount of feature engineering for them to be potentially useful:
- `earliest_cr_line`
- `last_credit_pull_d`

Since these date features require some feature engineering for modeling purposes, let's remove these date columns from the Dataframe.

## First 5 categorical columns
Let's explore the unique value counts of the columnns that seem like they contain categorical values.


```python
cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']
for col in cols:
    print(loans[col].value_counts(), end="\n-----\n")
    
```

    RENT        18112
    MORTGAGE    16686
    OWN          2778
    OTHER          96
    NONE            3
    Name: home_ownership, dtype: int64
    -----
    Not Verified       16281
    Verified           11856
    Source Verified     9538
    Name: verification_status, dtype: int64
    -----
    10+ years    8545
    < 1 year     4513
    2 years      4303
    3 years      4022
    4 years      3353
    5 years      3202
    1 year       3176
    6 years      2177
    7 years      1714
    8 years      1442
    9 years      1228
    Name: emp_length, dtype: int64
    -----
     36 months    28234
     60 months     9441
    Name: term, dtype: int64
    -----
    CA    6776
    NY    3614
    FL    2704
    TX    2613
    NJ    1776
    IL    1447
    PA    1442
    VA    1347
    GA    1323
    MA    1272
    OH    1149
    MD    1008
    AZ     807
    WA     788
    CO     748
    NC     729
    CT     711
    MI     678
    MO     648
    MN     581
    NV     466
    SC     454
    WI     427
    OR     422
    LA     420
    AL     420
    KY     311
    OK     285
    KS     249
    UT     249
    AR     229
    DC     209
    RI     194
    NM     180
    WV     164
    HI     162
    NH     157
    DE     110
    MT      77
    AK      76
    WY      76
    SD      60
    VT      53
    MS      19
    TN      17
    IN       9
    ID       6
    IA       5
    NE       5
    ME       3
    Name: addr_state, dtype: int64
    -----
    

The `home_ownership`, `verification_status`, `emp_length`, `term`, and `addr_state` columns all contain multiple discrete values. We should clean the `emp_length` column and treat it as a numerical one since the values have ordering (2 years of employment is less than 8 years).

Let's look at the unique value counts for the purpose and title columns


```python
print(loans["purpose"].value_counts())
print("----")
print(loans["title"].value_counts())

```

    debt_consolidation    17751
    credit_card            4911
    other                  3711
    home_improvement       2808
    major_purchase         2083
    small_business         1719
    car                    1459
    wedding                 916
    medical                 655
    moving                  552
    house                   356
    vacation                348
    educational             312
    renewable_energy         94
    Name: purpose, dtype: int64
    ----
    Debt Consolidation                                       2068
    Debt Consolidation Loan                                  1599
    Personal Loan                                             624
    Consolidation                                             488
    debt consolidation                                        466
    Credit Card Consolidation                                 345
    Home Improvement                                          336
    Debt consolidation                                        314
    Small Business Loan                                       298
    Credit Card Loan                                          294
    Personal                                                  290
    Consolidation Loan                                        250
    Home Improvement Loan                                     228
    personal loan                                             219
    Loan                                                      202
    Wedding Loan                                              199
    personal                                                  198
    Car Loan                                                  188
    consolidation                                             186
    Other Loan                                                168
    Wedding                                                   148
    Credit Card Payoff                                        144
    Credit Card Refinance                                     140
    Major Purchase Loan                                       131
    Consolidate                                               124
    Medical                                                   111
    Credit Card                                               110
    home improvement                                          101
    Credit Cards                                               91
    My Loan                                                    90
                                                             ... 
    uniquesites                                                 1
    Cleaning out the Credit Cards                               1
    Credit card dept consolidation                              1
    Haiti Relief funding for t-shirts                           1
    race trailer                                                1
    6/2/2011                                                    1
    The Trees                                                   1
    wheels                                                      1
    trying to pay of my bill!!!                                 1
    Try #2                                                      1
    CONSOLIDATE DEBTS                                           1
    Coin-Op Billiard                                            1
    Personal/ Consolidation                                     1
    Take the interest away from my Credit Card companies!       1
    honest borrower                                             1
    Debt Consolidate 2010                                       1
    Finish the Job                                              1
    Need better rate!!                                          1
    CREDIT CARD RELIEF                                          1
    get ahead of bills until July                               1
    Dream Wedding for my lil girl                               1
    private owner car purchase                                  1
    Cosmetic Surgery Loan                                       1
    Kangen Water Appliance                                      1
    MIA                                                         1
    startup home business Mary Kay                              1
    Resident Physician Moving Loan                              1
    HOme Improvement deer Creek road                            1
    Last semester of grad school                                1
    Personal Emergency                                          1
    Name: title, Length: 18881, dtype: int64
    

It seems like the `purpose` and `title` columns do contain overlapping information but we'll keep the `purpose` column since it contains a few discrete values. In addition, the `title` column has data quality issues since many of the values are repeated with slight modifications (e.g. `Debt Consolidation` and `Debt Consolidation Loan` and `debt consolidation`).

Remove the `last_credit_pull_d`, `addr_state`, `title`, and `earliest_cr_line` columns from loans.

The `home_ownership`, `verification_status`, `emp_length`, and `term` columns each contain a few discrete categorical values. We should encode these columns as dummy variables and keep them.

Convert the `int_rate` and `revol_util` columns to float columns.




```python
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loans = loans.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)

loans['int_rate'] = loans['int_rate'].str.rstrip('%').astype('float')

loans['revol_util'] = loans['revol_util'].str.rstrip('%').astype('float')

loans= loans.replace(mapping_dict)
```

Let's now encode the `home_ownership`, `verification_status`, `purpose`, and `term` columns as dummy variables and then drop the original columns entirely so we can use them in our model.


```python
cat_columns = ["home_ownership", "verification_status", "purpose", "term"]

dummy_df = pd.get_dummies(loans[cat_columns])

loans = pd.concat([loans, dummy_df], axis=1)

loans = loans.drop(cat_columns, axis=1)
```


```python
loans.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>emp_length</th>
      <th>annual_inc</th>
      <th>loan_status</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>...</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>term_ 36 months</th>
      <th>term_ 60 months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>10</td>
      <td>24000.0</td>
      <td>1</td>
      <td>27.65</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>0</td>
      <td>30000.0</td>
      <td>0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>10</td>
      <td>12252.0</td>
      <td>1</td>
      <td>8.72</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>10</td>
      <td>49200.0</td>
      <td>1</td>
      <td>20.00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5000.0</td>
      <td>7.90</td>
      <td>156.46</td>
      <td>3</td>
      <td>36000.0</td>
      <td>1</td>
      <td>11.20</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
loans.to_csv("cleaned_loans_2007.csv", index=False)
```


```python
cleaned_loans = pd.read_csv("cleaned_loans_2007.csv")

print(cleaned_loans.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37675 entries, 0 to 37674
    Data columns (total 38 columns):
    loan_amnt                              37675 non-null float64
    int_rate                               37675 non-null float64
    installment                            37675 non-null float64
    emp_length                             37675 non-null int64
    annual_inc                             37675 non-null float64
    loan_status                            37675 non-null int64
    dti                                    37675 non-null float64
    delinq_2yrs                            37675 non-null float64
    inq_last_6mths                         37675 non-null float64
    open_acc                               37675 non-null float64
    pub_rec                                37675 non-null float64
    revol_bal                              37675 non-null float64
    revol_util                             37675 non-null float64
    total_acc                              37675 non-null float64
    home_ownership_MORTGAGE                37675 non-null int64
    home_ownership_NONE                    37675 non-null int64
    home_ownership_OTHER                   37675 non-null int64
    home_ownership_OWN                     37675 non-null int64
    home_ownership_RENT                    37675 non-null int64
    verification_status_Not Verified       37675 non-null int64
    verification_status_Source Verified    37675 non-null int64
    verification_status_Verified           37675 non-null int64
    purpose_car                            37675 non-null int64
    purpose_credit_card                    37675 non-null int64
    purpose_debt_consolidation             37675 non-null int64
    purpose_educational                    37675 non-null int64
    purpose_home_improvement               37675 non-null int64
    purpose_house                          37675 non-null int64
    purpose_major_purchase                 37675 non-null int64
    purpose_medical                        37675 non-null int64
    purpose_moving                         37675 non-null int64
    purpose_other                          37675 non-null int64
    purpose_renewable_energy               37675 non-null int64
    purpose_small_business                 37675 non-null int64
    purpose_vacation                       37675 non-null int64
    purpose_wedding                        37675 non-null int64
    term_ 36 months                        37675 non-null int64
    term_ 60 months                        37675 non-null int64
    dtypes: float64(12), int64(26)
    memory usage: 10.9 MB
    None
    
