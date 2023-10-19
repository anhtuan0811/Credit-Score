# 🛒 Case Study - Credit Score Classification
<p align="right"> Using Python - Google Colab </p>


## :books: Table of Contents <!-- omit in toc -->

- [🔢 PYTHON - GOOGLE COLAB](#-python---google-colab)
  - [Import Required Library](#-Import-Required-)
  - [Data Cleaning](#-Data-Cleaning)
  - [Fitting](#fitting)
  - [Evaluate Model](#-evaluate-models)
 

---

## 👩🏼‍💻 PYTHON - GOOGLE COLAB

### 1️⃣ IMPORT REQUIRED LIBRARIES

<details><summary> Click to expand code </summary>
  
```python
#Import Library 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

```

```python
#mport dataset
df = pd.read_csv("train.csv")
df.head()
```
  
</details>

---
### 2️⃣ Data Cleaning

- There are 3 things that i would to do in this step:
  - The Overall Infomation
  - Alternative to info()
  - Outliers function with ( IQR_standard deviation)

<details><summary> 1.1 The  Overall Infomation </summary>

<br> We would check and clean the duplicated values of all columns, beside that we also drop some unnecessary columns.

```python
df.duplicated().sum()
df.shape
df.drop(['ID','Customer_ID','Name','SSN','Type_of_Loan'], axis=1, inplace=True)
df.columns
```
![image](https://github.com/anhtuan0811/Credit-Score/assets/143471832/f982d0a6-abca-4515-98bb-cacd52eacc97)

</details>

<details><summary> 1.2. Alternative to info() </summary>  

```python
def columns_info (df):
    columns=[]
    dtypes=[]
    unique=[]
    nunique=[]
    nulls=[]

    for colm in df.columns:
        columns.append(colm)
        dtypes.append(df[colm].dtypes)
        unique.append(df[colm].unique())
        nunique.append(df[colm].nunique())
        nulls.append(df[colm].isna().sum())

    return pd.DataFrame({'Columns':columns ,
                         'Data types':dtypes ,
                         'Unique values':unique ,
                         'Number of unique':nunique ,
                         'Missing Values':nulls
                          })

columns_info(df)

```
![image](https://github.com/anhtuan0811/Credit-Score/assets/143471832/d9d9dda2-9026-4444-bddc-1c5cbaeffa41)

<br>
  --> We can see the data types, unique values, number of unique values and null values, so we have to handle this.

</details>

<details><summary> 1.3. Outliers function with IQR_Standard Deviation </summary>  
  
```python
def check_outliers(colm,df):
    q1=df[colm].quantile(0.25)
    q3=df[colm].quantile(0.75)
    iqr=q3 - q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    outliers=[]

    for i in range(len(df)):
        value = df.loc[i,colm]
        if value > upper_bound or value < lower_bound:
            outliers.append(value)
    return outliers

def handle_outliers(colm,df):
    q1=df[colm].quantile(0.25)
    q3=df[colm].quantile(0.75)
    iqr=q3 - q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    outliers = []

    for i in range(len(df)):

        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm] = upper_bound
        elif df.loc[i,colm] < lower_bound:
            df.loc[i,colm] = lower_bound
```
<br> =>> 1.Check outliers: 

It calculates values related to the distribution of that column, including quartile 1 (q1), quartile 3 (q3), and the interquartile range (IQR).
Then, it computes the lower bound (lower_bound) and upper bound (upper_bound) based on the IQR.
Finally, it iterates through each row in the column and checks if the value of that row falls outside the lower bound or upper bound. If so, it records that value in the outliers list and returns this list.

=>> 2.Handling outliers:

It performs a similar calculation for q1, q3, IQR, lower_bound, and upper_bound.
Then, it iterates through each row in the column. If the value of that row is greater than the upper bound (upper_bound), it truncates the value to the upper bound. If the value of that row is less than the lower bound (lower_bound), it truncates the value to the lower bound.
This has the effect of reducing or eliminating outliers from the data, making the data more stable and less affected by outlier values.

</details>

<details><summary> 1.4. Result after cleaning data </summary>

<br> Regarding null values, we can replace them with the mean or median value, and for handling outlier values, I use the IQR method for removal

```python
df.info()
```

![image](https://github.com/anhtuan0811/Credit-Score/assets/143471832/55b9c3db-5a03-4c1a-8d03-3b46bfc15c32)
</details>

---
### 3️⃣ Fitting Model
<details><summary> Splitting Dataset  </summary> 
<br>
 
```python
X = df.drop('Credit_Score',axis=1).values
y = df['Credit_Score'].values


# split dataset to test and training set (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=)

#Evaluate the model's performance on the test data by making predictions and computing various performance metrics.

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
def evaluate_model(X_test,y_test,model):
    y_pred = model.predict(X_test)
    #accuracy
    acc = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred,average='macro')
    precision = precision_score(y_test,y_pred,average='macro')
    f1 = f1_score(y_test,y_pred,average='macro')
    cm = confusion_matrix(y_test,y_pred)

    return pd.Series({'Accuracy':acc,'Recall':recall,'Precision':precision,'F1 Score':f1})

```
</details>
  
---  
### 4️⃣ Evaluate Models

<details><summary> Random Forest </summary>

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

pd.DataFrame({'Random Forest Classifier (Test) ':evaluate_model(X_test,y_test,model),
              'Random Forest Classifier (Train)':evaluate_model(X_train,y_train,model)})
```
![image](https://github.com/anhtuan0811/Credit-Score/assets/143471832/c0725246-0384-40f3-8af5-2bcd7094c165)

</details>
<details><summary> KNN </summary>

```python
#Scale Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)

pd.DataFrame({'KNN (Test) ':evaluate_model(X_test,y_test,model2),
              'KNN (Train)':evaluate_model(X_train,y_train,model2)})
```
![image](https://github.com/anhtuan0811/Credit-Score/assets/143471832/6ad3c00a-333d-4f4b-b90f-dd2142a3b6c8)
