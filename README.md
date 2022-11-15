# Telco-Churn
Classification Project to predict Telco Customer churn
**Goal: Predicting Customer Churn for Vodafone

Hypothesis 1.Increases in monthly charges causes customers to churn 2.Customers with multiple lines are less likely to churn 3.Tech support for customers reduces churn rate for categories such as females and senior citizens

Questions

what is the chrun rate by: i. Payment method ii. Gender iii. Patner Status iv. Number of Dependents
Does increase in monthly charges influence: a. churn rate by citizenship? b. churn rate paperless vs. non- paperless?
Which internet service customers churn the most?
Does tech support influence customer churn in any way and by how much?
Importing Relevant Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer

# Set Training and Testing Data
from sklearn.model_selection import train_test_split
# Normalization
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
#Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#Model Evaluation
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, fbeta_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Testing Features
from sklearn.feature_selection import SelectFromModel
# Cross Validation and Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  RandomizedSearchCV

1.2 Data Cleaning and Preprocessing
churn rate was more as total charges increased but with monthly charges, increases did not significantly impact churn rate as seen by the distribution. Also with regards to tenure, the distribution of customers who churned most fell below the median indicating possibly that these might be fairly new customers or subscribers to the Telco service.

#churn rate by number of dependents
sns.countplot(data= data, x="Dependents", hue="Churn")
plt.title("Churn rate with respect to number of dependents")
plt.figure(figsize=(10,5))
plt.show()

<Figure size 720x360 with 0 Axes>
Observation: Customers without dependents churn more than customers with dependents

1.5.3.**Question 2: which Internet service customers churn the most?

#internet service users
Internet = data['InternetService']. value_counts()
Internet_customers =pd.DataFrame(Internet)
Internet_customers =Internet_customers.reset_index()
Internet_customers.columns =['InternetService','Total']
Internet_customers
InternetService	Total
0	Fiber optic	3096
1	DSL	2421
2	No	1526
#visualizing which internet service customers churn the most
sns.countplot(data= data, x="InternetService", hue="Churn")
plt.title("Churn rate with respect to Kind of Internet service")
plt.figure(figsize=(10,5))
plt.show()

<Figure size 720x360 with 0 Axes>
Observations: the table above indicates that most of the customers using internet service fall under the fiber optics category with a total of 3096 customers representing 43.95% of the whole. However from the plot they also are the category that churns most. To dig further, we try to find out if tech support has any influence on the churn rate trends depicted for fiber optic internet service users
Observations: the table above indicates that most of the customers using internet service fall under the fiber optics category with a total of 3096 customers representing 43.95% of the whole. However from the plot they also are the category that churns most. To dig further, we try to find out if tech support has any influence on the churn rate trends depicted for fiber optic internet service users
sns.set(rc={'figure.figsize':(10,5)})
sns.countplot(data= data, x="InternetService", hue="TechSupport", palette ="pastel")
plt.title("Churn rate with respect to Tech Support for Kind of Internet service")
Text(0.5, 1.0, 'Churn rate with respect to Tech Support for Kind of Internet service')

Observation: Customers using Fibre optic internet service did not receive tech support as much as customers using DSL internet service. A possible reason for the high churn rate amongst Fibre optic internet users as depicted earlier.

1.5.4.**Question 3: Does monthly charges influence churn rate by citizenship?

#  general preview 0f relationship between charges(total/monthly) and churn rate
sns.scatterplot(data =data, y='TotalCharges', x='MonthlyCharges', hue='Churn')
plt.title("Churn rate with respect to Charges")
Text(0.5, 1.0, 'Churn rate with respect to Charges')

Observation: There's a positive relationship between charges and churn rate. As charges increase(monthly/ Total), churnrate increases in proportion.

sns.set(rc={'figure.figsize':(10,5)})
sns.countplot(data= data, x="SeniorCitizen", hue="Churn")
plt.title("Churn rate with respect to Citizenship")
Text(0.5, 1.0, 'Churn rate with respect to Citizenship')

Observation: Non SeniorCitizens churned more than their counterpart Senior citizens

#visualizing if tech support had any influence on churn rate of seniorcitizens
sns.set(rc={'figure.figsize':(10,5)})
sns.countplot(data= data, x="SeniorCitizen", hue="TechSupport", palette ="pastel")
plt.title("Churn rate with respect to  Citizenship based on TechSupport")
Text(0.5, 1.0, 'Churn rate with respect to  Citizenship based on TechSupport')

Observation: Non Senior Citizens received TechSupport more than senior citizens but that did not positively influence their churn rate since they still churned the most.

**Hypothesis:Therefore, my hypothesis that Tech support for customers reduces churn rate for categories such as senior citizens is wrong and should be rejected.

#visualizing effect of  monthly charges on Citizenship churn
sns.barplot(data=data, x="SeniorCitizen", y="MonthlyCharges", hue ="Churn")
plt.title("Monthly charges on Citizenship Churn")
Text(0.5, 1.0, 'Monthly charges on Citizenship Churn')

Observation: Churn rates are higher for senior citizens when monthly charges go up and in comparison with non senior citizens the same is true but at a higher level.

1.5.5.**Question 4:Does Monthly Charges influence churn rate of customers with paperless billing or not

sns.barplot(data =data, y='MonthlyCharges', x='PaperlessBilling',hue ='Churn', palette = 'pastel')
plt.title("Churn rate with respect to Charges Vrs PaperlessBilling")
Text(0.5, 1.0, 'Churn rate with respect to Charges Vrs PaperlessBilling')

Observation: Paperless billing customers churned more as monthly charges increase

1.6.**Feature Engineering and Preprocessing

data.nunique()
​
​
gender                 2
SeniorCitizen          2
Partner                2
Dependents             2
tenure                73
PhoneService           2
MultipleLines          3
InternetService        3
OnlineSecurity         3
OnlineBackup           3
DeviceProtection       3
TechSupport            3
StreamingTV            3
StreamingMovies        3
Contract               3
PaperlessBilling       2
PaymentMethod          4
MonthlyCharges      1585
TotalCharges        6531
Churn                  2
dtype: int64
#One hot encoding categorical data
data = pd.get_dummies(data= data, columns=['MultipleLines','InternetService','OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod'])
data
gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	PaperlessBilling	MonthlyCharges	TotalCharges	Churn	...	StreamingMovies_0	StreamingMovies_1	StreamingMovies_No internet service	Contract_Month-to-month	Contract_One year	Contract_Two year	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check
0	1	0	0	1	1	1	0	29.85	29.85	1	...	0	1	0	1	0	0	0	0	1	0
1	0	0	1	1	34	0	1	56.95	1889.50	1	...	0	1	0	0	1	0	0	0	0	1
2	0	0	1	1	2	0	0	53.85	108.15	0	...	0	1	0	1	0	0	0	0	0	1
3	0	0	1	1	45	1	1	42.30	1840.75	1	...	0	1	0	0	1	0	1	0	0	0
4	1	0	1	1	2	0	0	70.70	151.65	0	...	0	1	0	1	0	0	0	0	1	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
7038	0	0	0	0	24	0	0	84.80	1990.50	1	...	1	0	0	0	1	0	0	0	0	1
7039	1	0	0	0	72	0	0	103.20	7362.90	1	...	1	0	0	0	1	0	0	1	0	0
7040	1	0	0	0	11	1	0	29.60	346.45	1	...	0	1	0	1	0	0	0	0	1	0
7041	0	1	0	1	4	0	0	74.40	306.60	0	...	0	1	0	1	0	0	0	0	0	1
7042	0	0	1	1	66	0	0	105.65	6844.50	1	...	1	0	0	0	0	1	1	0	0	0
7043 rows × 41 columns

1.6.1.Split data into Predictor and Response Variable

# Separate input features and target
X= data.drop(['Churn'], axis=1, inplace =False)
​
# Select Target
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
​
# Show the Training and Testing Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)
Shape of training feature: (5282, 40)
Shape of testing feature: (1761, 40)
Shape of training label: (5282,)
Shape of training label: (1761,)
1.6.2 Normalization

#Normalization
scaler = MinMaxScaler()
data_transform =['tenure', 'MonthlyCharges', 'TotalCharges']
data[data_transform]=scaler.fit_transform(data[data_transform])
data
gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	PaperlessBilling	MonthlyCharges	TotalCharges	Churn	...	StreamingMovies_0	StreamingMovies_1	StreamingMovies_No internet service	Contract_Month-to-month	Contract_One year	Contract_Two year	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check
0	1	0	0	1	0.013889	1	0	0.115423	0.001275	1	...	0	1	0	1	0	0	0	0	1	0
1	0	0	1	1	0.472222	0	1	0.385075	0.215867	1	...	0	1	0	0	1	0	0	0	0	1
2	0	0	1	1	0.027778	0	0	0.354229	0.010310	0	...	0	1	0	1	0	0	0	0	0	1
3	0	0	1	1	0.625000	1	1	0.239303	0.210241	1	...	0	1	0	0	1	0	1	0	0	0
4	1	0	1	1	0.027778	0	0	0.521891	0.015330	0	...	0	1	0	1	0	0	0	0	1	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
7038	0	0	0	0	0.333333	0	0	0.662189	0.227521	1	...	1	0	0	0	1	0	0	0	0	1
7039	1	0	0	0	1.000000	0	0	0.845274	0.847461	1	...	1	0	0	0	1	0	0	1	0	0
7040	1	0	0	0	0.152778	1	0	0.112935	0.037809	1	...	0	1	0	1	0	0	0	0	1	0
7041	0	1	0	1	0.055556	0	0	0.558706	0.033210	0	...	0	1	0	1	0	0	0	0	0	1
7042	0	0	1	1	0.916667	0	0	0.869652	0.787641	1	...	1	0	0	0	0	1	1	0	0	0
7043 rows × 41 columns

data.describe()
gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	PaperlessBilling	MonthlyCharges	TotalCharges	Churn	...	StreamingMovies_0	StreamingMovies_1	StreamingMovies_No internet service	Contract_Month-to-month	Contract_One year	Contract_Two year	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check
count	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	...	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000	7043.000000
mean	0.495244	0.162147	0.516967	0.700412	0.449599	0.096834	0.407781	0.462803	0.261309	0.734630	...	0.387903	0.395428	0.216669	0.550192	0.209144	0.240664	0.219225	0.216101	0.335794	0.228880
std	0.500013	0.368612	0.499748	0.458110	0.341104	0.295752	0.491457	0.299403	0.261366	0.441561	...	0.487307	0.488977	0.412004	0.497510	0.406726	0.427517	0.413751	0.411613	0.472301	0.420141
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	0.000000	0.000000	0.000000	0.000000	0.125000	0.000000	0.000000	0.171642	0.044245	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
50%	0.000000	0.000000	1.000000	1.000000	0.402778	0.000000	0.000000	0.518408	0.159445	1.000000	...	0.000000	0.000000	0.000000	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
75%	1.000000	0.000000	1.000000	1.000000	0.763889	0.000000	1.000000	0.712438	0.434780	1.000000	...	1.000000	1.000000	0.000000	1.000000	0.000000	0.000000	0.000000	0.000000	1.000000	0.000000
max	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	...	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000
8 rows × 41 columns

1.6.3.Check for Class Imbalance

labels =['Churn','No_Churn']
sns.color_palette()
sns.countplot(data.Churn)
sns.set(rc={'figure.figsize':(8,8)})

1.6.4.Perform SMOTE Sampling

sm = SMOTE(random_state=27, sampling_strategy=1.0)
X_train, y_train = sm.fit_resample(X_train, y_train)
np.unique(y_train, return_counts =True)
(array([0, 1], dtype=int64), array([3900, 3900], dtype=int64))
1.6.5. Model Building with (SMOTE) Balancing

1.Decision Tree

#joining the trainset for modelling
train_set = X_train.join(y_train, on = X_train.index)
train_set.head()
gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	PaperlessBilling	MonthlyCharges	TotalCharges	MultipleLines_No	...	StreamingMovies_1	StreamingMovies_No internet service	Contract_Month-to-month	Contract_One year	Contract_Two year	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check	Churn
0	1	0	1	0	1	0	1	59.20	59.2	1	...	0	0	1	0	0	0	0	1	0	0
1	0	0	1	1	29	0	0	58.75	1696.2	1	...	1	0	0	1	0	0	0	1	0	1
2	1	0	0	1	72	1	1	65.50	4919.7	0	...	0	0	0	0	1	1	0	0	0	1
3	1	0	1	1	23	0	0	20.30	470.6	1	...	0	1	1	0	0	1	0	0	0	1
4	1	0	0	1	72	0	0	92.40	6786.1	0	...	0	0	0	0	1	0	0	1	0	1
5 rows × 41 columns

#joining the testset for modelling
test_set = X_test.join(y_test, on = X_test.index)
test_set.head()
gender	SeniorCitizen	Partner	Dependents	tenure	PhoneService	PaperlessBilling	MonthlyCharges	TotalCharges	MultipleLines_No	...	StreamingMovies_1	StreamingMovies_No internet service	Contract_Month-to-month	Contract_One year	Contract_Two year	PaymentMethod_Bank transfer (automatic)	PaymentMethod_Credit card (automatic)	PaymentMethod_Electronic check	PaymentMethod_Mailed check	Churn
4903	1	0	1	1	5	0	0	90.80	455.50	0	...	1	0	1	0	0	0	0	1	0	0
2695	0	0	0	1	52	0	0	81.40	4354.45	0	...	0	0	0	1	0	0	1	0	0	1
2184	1	1	1	1	2	0	0	88.55	179.25	0	...	0	0	1	0	0	0	0	1	0	0
6024	1	0	0	1	3	0	0	91.50	242.95	0	...	0	0	1	0	0	0	1	0	0	0
5861	0	0	1	1	35	0	1	19.25	677.90	1	...	0	1	0	0	1	0	0	0	1	1
5 rows × 41 columns

# Separate input features and target
X= train_set.drop(['Churn'], axis=1, inplace =False)
​
# Select Target
y = train_set['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
​
dt =DecisionTreeClassifier()
dt.fit(X_train, y_train)

DecisionTreeClassifier
DecisionTreeClassifier()
y_pred= dt.predict(X_test)
# evaluating the model
acc=accuracy_score(y_test,y_pred)
prec= precision_score(y_test,y_pred)
rec= recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
results = pd.DataFrame([['Decision Tree',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(results)
           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  Decision Tree  0.808718   0.832988  0.791133  0.811521  0.799164
dtree = DecisionTreeClassifier(max_depth =10, random_state = 101, max_features= None, min_samples_leaf = 15)
dtree_smote =dtree.fit(X_train, y_train)
dtree_pred = dtree_smote.predict(X_test)
dtree_true = y_test
#visualizing Confusion Matrix
cm_dtree = confusion_matrix(dtree_true, dtree_pred)
f,ax = plt.subplots(figsize =(8,8))
sns.heatmap(cm_dtree, annot =True, linewidth =0.5, fmt=".0f",cmap ='RdPu', ax =ax)
plt.xlabel = ('dtree_pred')
plt.ylabel =('dtree_true')
plt.show()

dt =DecisionTreeClassifier()
dt.fit(X_train, y_train)
print('Feature Importances:',dt.feature_importances_)
​
​
Feature Importances: [1.76299227e-02 1.57756275e-02 1.03230683e-02 1.52053909e-02
 8.16483438e-02 2.59644436e-03 5.06031040e-02 1.27293445e-01
 1.24643794e-01 1.07465241e-02 1.16593295e-03 9.69452913e-03
 6.85107333e-03 5.89152020e-03 0.00000000e+00 2.19928320e-02
 4.14118155e-03 2.90567036e-05 5.43103236e-03 0.00000000e+00
 1.11794481e-02 5.67985912e-03 7.57478281e-03 0.00000000e+00
 3.04146266e-02 2.55030379e-03 0.00000000e+00 4.76024060e-03
 7.44223096e-03 1.28226880e-02 8.10170950e-03 9.67763793e-03
 1.00818908e-03 1.82848265e-01 1.04228868e-01 4.00024130e-02
 1.50408219e-02 1.71072626e-02 1.37335675e-02 1.41642629e-02]
#plot feature importance
​
feature_importance = dt.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')
​
Text(0.5, 1.0, 'Feature Importance')

2.Logistic Regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

LogisticRegression
LogisticRegression()
#predicting test results and calculating accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
Accuracy of logistic regression classifier on test set: 0.82
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.79      0.85      0.82      1133
           1       0.85      0.79      0.82      1207

    accuracy                           0.82      2340
   macro avg       0.82      0.82      0.82      2340
weighted avg       0.82      0.82      0.82      2340

3.Gradient Boosting

xgc =xgb.XGBClassifier(n_estimators = 500, max_depth =5, random_state =42)
xgc.fit(X_train, y_train)

XGBClassifier
XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_bin=256, max_cat_threshold=64, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
              missing=nan, monotone_constraints='()', n_estimators=500,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=42, ...)
#predicting the testset results
y_pred =xgc.predict(X_test)
# evaluating the model
acc=accuracy_score(y_test,y_pred)
prec= precision_score(y_test,y_pred)
rec= recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
results = pd.DataFrame([['XGBClassifier',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(results)
           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  XGBClassifier  0.831624   0.840738  0.830986  0.835833  0.832918
4.Support Vector Machines

SVC =SVC(kernel ='rbf')
SVC.fit(X_train,y_train)

SVC
SVC()
#predicting the testset results
y_pred =SVC.predict(X_test)
# evaluating the model
acc=accuracy_score(y_test,y_pred)
prec= precision_score(y_test,y_pred)
rec= recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
results = pd.DataFrame([['SVC',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(results)
  Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0   SVC  0.641026   0.641699  0.688484  0.664269  0.678589
5.Random Forest

rf_clf = RandomForestClassifier()   
rf_clf.fit(X_train,y_train)

RandomForestClassifier
RandomForestClassifier()
#predicting the testset results
y_pred =rf_clf.predict(X_test)
# evaluating the model
acc=accuracy_score(y_test,y_pred)
prec= precision_score(y_test,y_pred)
rec= recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
results = pd.DataFrame([['Random Forest',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(results)
​
           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  Random Forest  0.849145    0.87066  0.830986   0.85036  0.838629
1.6.7.Summarizing the performance of models

Observations: From the five models(SMOTE) performed to train the model, the Random Forest and Xgboost models have the highest F1 score of 85.2% and 82.5% respectivelyand since higher F1 scores are generally better, we go ahead to do some iterations to choose the best model.

1.6.8.** Iteration using Important Features

# evaluating model by using important features
# first visualize important features
xgc =xgb.XGBClassifier(random_state =42)
xgc.fit(X_train, y_train)
print('Feature Importances:',xgc.feature_importances_)
Feature Importances: [0.00390246 0.00548948 0.0042281  0.00314573 0.01006642 0.02047577
 0.02420156 0.00469362 0.0043467  0.01990519 0.         0.01350805
 0.03418499 0.09706294 0.03722137 0.01445039 0.01035724 0.
 0.01169112 0.         0.01420394 0.00673506 0.00545864 0.
 0.01876104 0.01313043 0.         0.00638323 0.00716402 0.
 0.00812934 0.00817071 0.         0.22317994 0.1990236  0.10151263
 0.01541373 0.02647745 0.00921626 0.0181088 ]
#plot feature importance
feature_importance = xgc.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title('Feature Importance')
Text(0.5, 1.0, 'Feature Importance')

selection=SelectFromModel(xgc)
selection.fit(X_train, y_train)
SelectFromModel
estimator: XGBClassifier

XGBClassifier
# transform the train and test features
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)
​
#train model
xgc.fit(select_X_train, y_train)

XGBClassifier
XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_bin=256, max_cat_threshold=64, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
              missing=nan, monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=42, ...)
#predicting test results
y_pred = xgc.predict(select_X_test)
​
#evaluating the model
xgc_acc=accuracy_score(y_test,y_pred)
xgc_fscore = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
​
print('Limited Features XGBoost Model Accuracy:', xgc_acc)
print('Limited Features XGBoost Model F1 Score:', xgc_fscore)
print('Limited Features XGBoost Model F2 Score:', f2)
Limited Features XGBoost Model Accuracy: 0.7713675213675214
Limited Features XGBoost Model F1 Score: 0.7571493418066273
Limited Features XGBoost Model F2 Score: 0.7160027472527472
​
selection=SelectFromModel(rf_clf)
selection.fit(X_train, y_train)
SelectFromModel
estimator: RandomForestClassifier

RandomForestClassifier
# transform the train and test features
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)
​
#train model
rf_clf.fit(select_X_train, y_train)

RandomForestClassifier
RandomForestClassifier()
#predicting test results
y_pred = rf_clf.predict(select_X_test)
​
#evaluating the model
rf_clf_acc=accuracy_score(y_test,y_pred)
rf_clf_fscore = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
​
print('Limited Features RandomForest Model Accuracy:', rf_clf_acc)
print('Limited Features RandomForest Model F1 Score:', rf_clf_fscore)
print('Limited Features RandomForest Model F2 Score:', f2)
Limited Features RandomForest Model Accuracy: 0.826068376068376
Limited Features RandomForest Model F1 Score: 0.8279069767441861
Limited Features RandomForest Model F2 Score: 0.8177413965920481
Notes: the first model without the limited features for both XGBoost and RandomForest models perform better that with limited features therefore we stick to the first models

2.0. **Model Buidling without Smote Balancing

# Separate input features and target
wsb_X= train_set.drop(['Churn'], axis=1, inplace =False)
​
# Select Target
wsb_y = train_set['Churn']
​
wsb_X_train, wsb_X_test, wsb_y_train, wsb_y_test = train_test_split(wsb_X, wsb_y, test_size=0.25, random_state=27)
i.Decision Tree Model

dt =DecisionTreeClassifier()
dt.fit(wsb_X_train, wsb_y_train)
wsb_y_pred= dt.predict(wsb_X_test)
# evaluating the model
acc=accuracy_score(wsb_y_test,wsb_y_pred)
prec= precision_score(wsb_y_test,wsb_y_pred)
rec= recall_score(wsb_y_test,wsb_y_pred)
f1 = f1_score(wsb_y_test,wsb_y_pred)
f2 = fbeta_score(wsb_y_test,wsb_y_pred, beta =2.0)
dt_results_wsb = pd.DataFrame([['Decision Tree',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(dt_results_wsb)
           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  Decision Tree  0.804615   0.824795  0.793103  0.808639  0.799245
ii.Logistic Regression Model

wsb_X_train, wsb_X_test, wsb_y_train, wsb_y_test = train_test_split(wsb_X, wsb_y, test_size=0.3, random_state=27)
logreg = LogisticRegression()
logreg.fit(wsb_X_train, wsb_y_train)

LogisticRegression
LogisticRegression()
#predicting test results and calculating accuracy
wsb_y_pred = logreg.predict(wsb_X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(wsb_X_test, wsb_y_test)))
Accuracy of logistic regression classifier on test set: 0.84
print(classification_report(wsb_y_test, wsb_y_pred))
              precision    recall  f1-score   support

           0       0.82      0.86      0.84      1154
           1       0.86      0.81      0.83      1186

    accuracy                           0.84      2340
   macro avg       0.84      0.84      0.84      2340
weighted avg       0.84      0.84      0.84      2340

iii. Gradient Boosting

xgc =xgb.XGBClassifier(n_estimators = 500, max_depth =5, random_state =42)
xgc.fit(wsb_X_train, wsb_y_train)

XGBClassifier
XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_bin=256, max_cat_threshold=64, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
              missing=nan, monotone_constraints='()', n_estimators=500,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=42, ...)
#predicting the testset results
wsb_y_pred =xgc.predict(wsb_X_test)
​
# evaluating the model
acc=accuracy_score(wsb_y_test,wsb_y_pred)
prec= precision_score(wsb_y_test,wsb_y_pred)
rec= recall_score(wsb_y_test,wsb_y_pred)
f1 = f1_score(wsb_y_test,wsb_y_pred)
f2 = fbeta_score(wsb_y_test,wsb_y_pred, beta =2.0)
wsb_results2 = pd.DataFrame([['XGBClassifier',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(wsb_results2)
           Model  Accuracy  Presicion   Recall  F1 Score  F2 Score
0  XGBClassifier  0.846581    0.83977  0.86172  0.850603  0.857239
iv. Random Classifier Model

rf_clf = RandomForestClassifier()   
rf_clf.fit(wsb_X_train, wsb_y_train)

RandomForestClassifier
RandomForestClassifier()
#predicting the testset results
wsb_y_pred =rf_clf.predict(wsb_X_test)
# evaluating the model
acc=accuracy_score(wsb_y_test,wsb_y_pred)
prec= precision_score(wsb_y_test,wsb_y_pred)
rec= recall_score(wsb_y_test,wsb_y_pred)
f1 = f1_score(wsb_y_test,wsb_y_pred)
f2 = fbeta_score(wsb_y_test,wsb_y_pred, beta =2.0)
wsb_results3 = pd.DataFrame([['Random Forest',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(wsb_results3)
           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  Random Forest  0.857265   0.858586  0.860034  0.859309  0.859744
3.0. **Model Improvement
#for randomforestclassifier (SMOTE)
rf_cl =RandomForestClassifier(n_estimators =33)
rf_cl.fit(X_train, y_train)
#predicting test results
y_pred = rf_cl.predict(X_test)
# evaluating the model
acc=accuracy_score(y_test,y_pred)
prec= precision_score(y_test,y_pred)
rec= recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
results = pd.DataFrame([['Random Forest',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(results)
           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  Random Forest  0.850855   0.865417  0.841756  0.853423  0.846385
#for randomforestclassifier ( Without SMOTE)
rf_cl =RandomForestClassifier(n_estimators =33)
rf_cl.fit(wsb_X_train, wsb_y_train)
#predicting the testset results
wsb_y_pred =rf_clf.predict(wsb_X_test)
# evaluating the model
acc=accuracy_score(wsb_y_test,wsb_y_pred)
prec= precision_score(wsb_y_test,wsb_y_pred)
rec= recall_score(wsb_y_test,wsb_y_pred)
f1 = f1_score(wsb_y_test,wsb_y_pred)
f2 = fbeta_score(wsb_y_test,wsb_y_pred, beta =2.0)
wsb_results= pd.DataFrame([['Random Forest',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(wsb_results)
           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  Random Forest  0.857265   0.858586  0.860034  0.859309  0.859744
#for XGBClassifier(SMOTE)
xgc =xgb.XGBClassifier(n_estimators =500,max_depths =5, random_state=42)
xgc.fit(X_train, y_train)
#predicting test results
y_pred = xgc.predict(X_test)
# evaluating the model
acc=accuracy_score(y_test,y_pred)
prec= precision_score(y_test,y_pred)
rec= recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
f2 = fbeta_score(y_test,y_pred, beta =2.0)
results = pd.DataFrame([['XGBClassifier',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(results)

           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  XGBClassifier  0.832906   0.842857  0.830986  0.836879  0.833333
#for XGBClassifier(Without SMOTE)
xgc =xgb.XGBClassifier(n_estimators =500,max_depths =5, random_state=42)
xgc.fit(wsb_X_train, wsb_y_train)
​
#predicting the testset results
wsb_y_pred =xgc.predict(wsb_X_test)
​
# evaluating the model
acc=accuracy_score(wsb_y_test,wsb_y_pred)
prec= precision_score(wsb_y_test,wsb_y_pred)
rec= recall_score(wsb_y_test,wsb_y_pred)
f1 = f1_score(wsb_y_test,wsb_y_pred)
f2 = fbeta_score(wsb_y_test,wsb_y_pred, beta =2.0)
wsb_results2 = pd.DataFrame([['XGBClassifier',acc, prec, rec, f1, f2]], columns =['Model','Accuracy','Presicion','Recall','F1 Score', 'F2 Score'])
print(wsb_results2)
[01:40:01] WARNING: C:/buildkite-agent/builds/buildkite-windows-cpu-autoscaling-group-i-03de431ba26204c4d-1/xgboost/xgboost-ci-windows/src/learner.cc:767: 
Parameters: { "max_depths" } are not used.

           Model  Accuracy  Presicion    Recall  F1 Score  F2 Score
0  XGBClassifier  0.844872   0.839242  0.858347  0.848687  0.854457
4.0.**Evaluate Chosen model

4.1.Cross Validation and Hyperparameter Tuning

since XGboost and Random Forest Models without Smote perform better than models with smote, we perform cross validation and hyperparameter tuning on the selected without SMOTE models

accuracy_score(wsb_y_test, wsb_y_pred)
0.8448717948717949
#model on which to use CV without SMOTE
xgc =xgb.XGBClassifier(n_estimators =500, random_state =42)
#define cross-validation method
kfold =KFold(n_splits=5) 
#evaluate model
results =cross_val_score(xgc, wsb_X_train, wsb_y_train, cv=kfold)
​
scores = cross_val_score(xgc, wsb_X_train,wsb_y_train, cv=kfold, n_jobs=-1)
print(scores.mean())
0.8371794871794872
#model on which to use CV with SMOTE
xgc =xgb.XGBClassifier(n_estimators =500, random_state =42)
#define cross-validation method
kfold =KFold(n_splits=5) 
#evaluate model
results =cross_val_score(xgc, X_train, y_train, cv=kfold)
​
scores = cross_val_score(xgc, X_train,y_train, cv=kfold, n_jobs=-1)
print(scores.mean())
0.845970695970696
#model on which to use CV without Smote
rf_cl =RandomForestClassifier(n_estimators =500, random_state =42)
#define cross-validation method
kfold =KFold(n_splits=5) 
#evaluate model
results =cross_val_score(rf_cl, wsb_X_train, wsb_y_train, cv=kfold)
​
scores = cross_val_score(rf_cl, wsb_X_train,wsb_y_train, cv=kfold, n_jobs=-1)
print(scores.mean())
0.8507326007326007
#model on which to use CV with Smote
rf_cl =RandomForestClassifier(n_estimators =500, random_state =42)
#define cross-validation method
kfold =KFold(n_splits=5) 
#evaluate model
results =cross_val_score(rf_cl, X_train, y_train, cv=kfold)
​
scores = cross_val_score(rf_cl, X_train,y_train, cv=kfold, n_jobs=-1)
print(scores.mean())
0.8553113553113553
with cross validation, we note that the RandomClassifier model with Smote performs slightly better than Xgboost. From results, on average, we expect the XGboost model to be able to predict unkown data accurately at 84.5% while RandomForest is expected to predict at 85.53%

Conversely, cross validation on Random Classifier model without Smote is 85.07% and that of XGBoost model 83.7% which are both comparatively lower than the former though in the same range. Thus we do Hyperparameter tuning only for Models with Smote.

#Hyperparameter(Grid Search Method) tuning for Random Forest Classifier
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
{ 'n_estimators':np.arange(5,100,5), 'max_features':np.arange(0.1,1.0,0.05), }
,cv=5,verbose=1,n_jobs=-1 ) 
grid_search.fit(X_train,y_train)
Fitting 5 folds for each of 342 candidates, totalling 1710 fits
GridSearchCV
estimator: RandomForestClassifier

RandomForestClassifier
grid_search.best_params_
{'max_features': 0.15000000000000002, 'n_estimators': 90}
grid_search.best_score_
0.8569597069597069
#Hyperparameter(Grid Search Method) tuning for XGB Classifier
grid_search = GridSearchCV(xgb.XGBClassifier(random_state=42), 
{ 'n_estimators':np.arange(5,100,5), 'max_features':np.arange(0.1,1.0,0.05), }
,cv=5, verbose=1,n_jobs=-1 ) 
grid_search.fit(X_train,y_train)
Fitting 5 folds for each of 342 candidates, totalling 1710 fits
[02:15:19] WARNING: C:/buildkite-agent/builds/buildkite-windows-cpu-autoscaling-group-i-03de431ba26204c4d-1/xgboost/xgboost-ci-windows/src/learner.cc:767: 
Parameters: { "max_features" } are not used.

GridSearchCV
estimator: XGBClassifier

XGBClassifier
grid_search.best_params_
{'max_features': 0.1, 'n_estimators': 35}
grid_search.best_score_
0.8589743589743591
Model's best score for Random Classifier and XGB is 0.8569 and 0.8589 respectively using GridSearch method

# Hyperparameter tuning (Randomized search method)n forXGBClassifier
random_search = RandomizedSearchCV(xgb.XGBClassifier(random_state=42), 
{ 'n_estimators':np.arange(5,100,5), 'max_features':np.arange(0.1,1.0,0.05), }
,cv=5, verbose=1,n_jobs=-1, n_iter=50, random_state = 0 ) 
random_search.fit(X_train,y_train)
Fitting 5 folds for each of 50 candidates, totalling 250 fits
[02:18:34] WARNING: C:/buildkite-agent/builds/buildkite-windows-cpu-autoscaling-group-i-03de431ba26204c4d-1/xgboost/xgboost-ci-windows/src/learner.cc:767: 
Parameters: { "max_features" } are not used.

RandomizedSearchCV
estimator: XGBClassifier

XGBClassifier
random_search.best_params_
{'n_estimators': 35, 'max_features': 0.1}
random_search.best_score_
0.8589743589743591
# Hyperparameter tuning (Randomized search method) for RandomForest Classifier
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), 
{ 'n_estimators':np.arange(5,100,5), 'max_features':np.arange(0.1,1.0,0.05), }
,cv=5,verbose=1,n_jobs=-1, n_iter=50, random_state = 0 ) 
random_search.fit(X_train,y_train)
Fitting 5 folds for each of 50 candidates, totalling 250 fits
RandomizedSearchCV
estimator: RandomForestClassifier

RandomForestClassifier
random_search.best_params_
{'n_estimators': 95, 'max_features': 0.15000000000000002}
random_search.best_score_
0.8558608058608058
models best score for Randomclassifier model is 0.8558 and Model's best score for XGBClassifier is 0.8589 using Randomsearch Method

From both the Gridsearch and Randomized search,XGBoost model still performs a little higher than Randomclassifier model and so we settle on XGBoost to build an optimized version of the model using the combination of hyperparameters from the Randomizedsearch method.

5.0.** Future Prediction

#fitting the best model to the train data
best_XGBoost_model =random_search.fit(X_train,y_train)
#predicting the testset results
best_XGBoost_pred =best_XGBoost_model.predict(X_test)
#evaluating the model
print(classification_report(y_test, best_XGBoost_pred))
Fitting 5 folds for each of 50 candidates, totalling 250 fits
              precision    recall  f1-score   support

           0       0.83      0.87      0.85      1133
           1       0.87      0.84      0.85      1207

    accuracy                           0.85      2340
   macro avg       0.85      0.85      0.85      2340
weighted avg       0.85      0.85      0.85      2340

#visualizing Confusion Matrix
cm_best_XGBoost = confusion_matrix(y_test, best_XGBoost_pred)
f,ax = plt.subplots(figsize =(8,8))
sns.heatmap(cm_best_XGBoost, annot =True, linewidth =0.5, fmt=".0f",cmap ='RdPu', ax =ax)
plt.xlabel = ('best_XGBoost_pred')
plt.ylabel =('y_test')
plt.show()

#get final_cv_score
#define cross-validation method
kfold =KFold(n_splits=5) 
final_cv_score =cross_val_score(best_XGBoost_model, X_train, y_train, cv=kfold)
​
final_cv_score = cross_val_score(best_XGBoost_model, X_train,y_train, cv=kfold, n_jobs=-1)
print(final_cv_score.mean())
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Fitting 5 folds for each of 50 candidates, totalling 250 fits
0.8536630036630036
conclusion: From the scores and confusion matrix above, we see that this version of the model is the best for now with an average cross validation score of 85.36% and F1(0.84). We can expect that this model will reliably predict which customers are likely to churn and inform Vodafone what strategies to implement for customer retention
conclusion: From the scores and confusion matrix above, we see that this version of the model is the best for now with an average cross validation score of 85.36% and F1(0.84). We can expect that this model will reliably predict which customers are likely 
