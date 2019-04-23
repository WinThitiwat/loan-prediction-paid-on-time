
# Notes:

Check out `NOTEBOOK - Data Cleaning` to see cleaning and preparing dataset processes that contains data on loans made to members of [Lending Club](https://www.lendingclub.com)

Our eventual goal is to generate features from the data, which can feed into a machine learning algorithm. The algorithm will make predictions about whether or not a loan will be paid off on time, which is contained in the `loan_status` column of the clean dataset.

# Loading cleaned dataset


```python
import pandas as pd

loans = pd.read_csv("cleaned_loans_2007.csv")

print(loans.info())
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
    

# Selecting an Error Metric

**Can we build a machine learning model that can accurately predict if a borrower will pay off their loan on time or not?"**

| Actual Loan Status | Prediction | Error Type     |
| ------------------ |:----------:| --------------:|
|     0              | 1          | False Positive |
|     1              | 1          | True Positive  |
|     0              | 0          | True Negative  |
|     1              | 0          | False Negative |

With a false positive, we predict that a loan will be paid off on time, but it actually isn't. This costs us money, since we fund loans that lose us money. With a false negative, we predict that a loan won't be paid off on time, but it actually would be paid off on time. This loses us potential money, since we didn't fund a loan that actually would have been paid off

A conservative investor would want to minimize risk, and avoid funding a risky loan (false positives) as much as possible while they'd be more okay with missing out on opportunities (false negatives). So, in this case, we're primarily concerned with **false positives**.



# Class Imbalance

We have to take the `loan_status` column into account as well in terms of class imbalance. There are 6 times as many loans that were paid off on time (1), than loans that weren't paid off on time (0) [see the codes below where]


```python
loans["loan_status"].value_counts()
```




    1    32286
    0     5389
    Name: loan_status, dtype: int64



This causes a major issue when we use accuracy as a metric. This is because due to the class imbalance, a classifier can predict 1 for every row, and still have high accuracy. 

* Example to explain the issue:

| Actual Loan Status | Prediction | Profit/loss    |
| ------------------ |:----------:| --------------:|
|     0              | 1          |   -1000        |
|     1              | 1          | 100            |
|     1              | 1          | 100            |
|     1              | 1          | 100            |
|     1              | 1          | 100            |
|     1              | 1          | 100            |
|     1              | 1          | 100            |

Our predictions are `87.5%` accurate (# of correct prediction, which is 6 divided by total number of prediction, which is 7). We've correctly identified `loan_status` in `87.5%` of classes. 

However, even though out model is technically accurate, in which we made `600` dollars in interest from the borrowers that paid us back, we lost `1000` dollars on the borrower who never paid us back. As a result, we ended up losing `400` dollars overall.


# First Prediction - Logistic Regression

Let's try a good first algorithm to apply binary classification problem, which is `Logistic Regression` to give us an overview prediction for the current dataset.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

lr = LogisticRegression()

cols = loans.columns
train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]

predictions = cross_val_predict(lr, features, target, cv=3)

predictions = pd.Series(predictions)
# False positives.

fp_filter = (predictions==1)&(loans["loan_status"] ==0)
fp = len(predictions[fp_filter])

tp_filter = (predictions==1)&(loans["loan_status"] ==1)
tp = len(predictions[tp_filter])

fn_filter = (predictions==0)&(loans["loan_status"] ==1)
fn = len(predictions[fn_filter])

tn_filter = (predictions==0)&(loans["loan_status"] ==0)
tn = len(predictions[tn_filter])

tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

print( predictions.value_counts(), "\n----")

print("First 10 prediction result:", predictions.head(10))
```

    1    37616
    0       59
    dtype: int64 
    ----
    First 10 prediction result: 0    1
    1    1
    2    1
    3    1
    4    1
    5    1
    6    1
    7    1
    8    1
    9    1
    dtype: int64
    


```python
print("True Positive Rate: ", tpr)
print("False Positive Rate: ", fpr)
```

    True Positive Rate:  0.9987920460880877
    False Positive Rate:  0.9962887363147152
    

Our `fpr` and `tpr` are around what we'd expect if the model was predicting all ones as the target classes are imbalanced. 

Let's get a classifier to correct for imbalanced classes.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

lr = LogisticRegression(class_weight='balanced')

predictions = cross_val_predict(lr, features, target, cv=3)

predictions = pd.Series(predictions)
# False positives.

fp_filter = (predictions==1)&(loans["loan_status"] ==0)
fp = len(predictions[fp_filter])

tp_filter = (predictions==1)&(loans["loan_status"] ==1)
tp = len(predictions[tp_filter])

fn_filter = (predictions==0)&(loans["loan_status"] ==1)
fn = len(predictions[fn_filter])

tn_filter = (predictions==0)&(loans["loan_status"] ==0)
tn = len(predictions[tn_filter])

tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

print("True Positive Rate: ", tpr)
print("False Positive Rate: ", fpr)
```

    True Positive Rate:  0.6647463296784984
    False Positive Rate:  0.38040452774169603
    

We significantly improved false positive rate in the last screen by balancing the classes, which reduced true positive rate. Our true positive rate is now around `67%`, and our false positive rate is around `40%`.

From a conservative investor's standpoint, it's reassuring that the false positive rate is lower because it means that we'll be able to do a better job at avoiding bad loans than if we funded everything. However, we'd only ever decide to fund 67% of the total loans (true positive rate), so we'd immediately reject a good amount of loans.

Let's change the `class_weight` parameter from the string "balanced" to a dictionary of penalty values manually.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

penalty = {
    0:10,
    1:1
}
lr = LogisticRegression(class_weight=penalty)

predictions = cross_val_predict(lr, features, target, cv=3)

predictions = pd.Series(predictions)
# False positives.

fp_filter = (predictions==1)&(loans["loan_status"] ==0)
fp = len(predictions[fp_filter])

tp_filter = (predictions==1)&(loans["loan_status"] ==1)
tp = len(predictions[tp_filter])

fn_filter = (predictions==0)&(loans["loan_status"] ==1)
fn = len(predictions[fn_filter])

tn_filter = (predictions==0)&(loans["loan_status"] ==0)
tn = len(predictions[tn_filter])

tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

print("True Positive Rate: ", tpr)
print("False Positive Rate: ", fpr)
```

    True Positive Rate:  0.2475995787647897
    False Positive Rate:  0.09352384486917796
    

It looks like assigning manual penalties lowered the false positive rate to `7%`, and thus lowered our risk.

Note that this comes at the expense of true positive rate. While we have fewer false positives, we're also missing opportunities to fund more loans and potentially make more money. Given that we're approaching this as a conservative investor, this strategy makes sense, but it's worth keeping in mind the tradeoffs.

# Let's try a more complex algorithm, random forest 


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict



lr = RandomForestClassifier(class_weight="balanced", random_state=1)

predictions = cross_val_predict(lr, features, target, cv=3)

predictions = pd.Series(predictions)
# False positives.

fp_filter = (predictions==1)&(loans["loan_status"] ==0)
fp = len(predictions[fp_filter])

tp_filter = (predictions==1)&(loans["loan_status"] ==1)
tp = len(predictions[tp_filter])

fn_filter = (predictions==0)&(loans["loan_status"] ==1)
fn = len(predictions[fn_filter])

tn_filter = (predictions==0)&(loans["loan_status"] ==0)
tn = len(predictions[tn_filter])

tpr = tp/(tp+fn)
fpr = fp/(fp+tn)

print("True Positive Rate: ", tpr)
print("False Positive Rate: ", fpr)
```

    c:\python36\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

    True Positive Rate:  0.9700799107972495
    False Positive Rate:  0.9181666357394693
    

Unfortunately, using a random forest classifier didn't improve our false positive rate.The model is likely weighting too heavily on the `1` class, and still mostly predicting `1s`. 

Ultimately, our best model had a false positive rate of `7%`, and a true positive rate of `20%`. For a conservative investor, this means that they make money as long as the interest rate is high enough to offset the losses from `7%` of borrowers defaulting, and that the pool of `20%` of borrowers is large enough to make enough interest money to offset the losses.
