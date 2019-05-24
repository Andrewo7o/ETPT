import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

train = pd.read_csv('./pfm_train.csv')
test = pd.read_csv('./pfm_test.csv')

train.drop(['EmployeeNumber'], axis = 1, inplace = True)
Attrition = train['Attrition']
train.drop(['Attrition'], axis = 1, inplace = True)
train.insert(0, 'Attrition', Attrition)
train.drop(['Over18', 'StandardHours'], axis = 1, inplace = True)

train['MonthlyIncome'] = pd.cut(train['MonthlyIncome'], bins=10)
col_object = []
for col in train.columns[1:]:
    if train[col].dtype == 'object':
        col_object.append(col)
print(col_object)

train_encode = pd.get_dummies(train)
train_encode.drop(['TotalWorkingYears', 'YearsWithCurrManager'], axis = 1, inplace = True)

X = train_encode.iloc[:, 1:]
y = train_encode.iloc[:, 0]

'''
clf = RandomForestClassifier(n_estimators=50)
x_tra,x_val,y_tra,y_val = train_test_split(X, y, test_size=0.2)
clf.fit(x_tra,y_tra)

print (confusion_matrix(y_val,clf.predict(x_val)))
print (accuracy_score(y_val,clf.predict(x_val)))


'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))


pred = lr.predict(X_test)
print(np.mean(pred == y_test))


test.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis = 1, inplace = True)
test_MonthlyIncome = pd.concat((pd.Series([1009, 19999]), test['MonthlyIncome']))
test['MonthlyIncome'] = pd.cut(test_MonthlyIncome, bins=10)[2:] 

test_encode = pd.get_dummies(test)
test_encode.drop(['TotalWorkingYears', 'YearsWithCurrManager'], axis = 1, inplace = True)
sample = pd.DataFrame(lr.predict(test_encode))
sample.to_csv('sample.csv')
