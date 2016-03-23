
"""
Supervised Learning: Titanic

Analysis, Visualization and Testing Machine Learning on predicting survival rate of titanic members.

Dataset: belong to Kaggle official site
Cannot be distributed outside without allowance from Kaggle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


titanic = pd.read_csv("train.csv")
'''
tmp = titanic[titanic["Cabin"].isnull() == False]
tmp["Cabin"] = tmp["Cabin"].apply(lambda r: r[0])
pd.crosstab(tmp["Survived"],tmp["Cabin"]).mean()

# The ones who were assigned a Cabin letters(except T) have a greater chance to survive
# But since there's many missing Cabin assigned, we can drop "Cabin" feature from analysis

titanic["Cabin"].fillna(value="T", inplace=True)
titanic["Cabin"]=titanic.apply(lambda r: r[0])
titanic.loc[titanic["Cabin"] != 'T', "Cabin"] = 1
titanic.loc[titanic["Cabin"] == 'T', "Cabin"] = 0
'''
titanic = titanic.drop(["Cabin"], axis=1)

# Create family size and drop 2 columns SibSp and Parch
titanic["Family size"] = titanic["SibSp"] + titanic["Parch"]
titanic = titanic.drop(['SibSp','Parch', 'PassengerId', 'Ticket'], axis=1)

#==================VISUALIZE AND ANALYZE==========================================
# 1. EMBARKED AND SEX
titanic["Embarked"].value_counts()
# Fill Null values of Embarked with the most popular S
titanic["Embarked"] = titanic["Embarked"].fillna("S")
fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize=(15, 6))
sns.factorplot('Embarked','Survived', data=titanic, ax=axis1)
sns.factorplot('Embarked','Survived', hue= "Sex", kind = 'bar', data=titanic,ax=axis2, palette="PRGn")
sns.factorplot('Sex','Survived', data=titanic, ax=axis3)

# Embarked C has good rate for survival, as same as female gender
# Convert Male/Female to 0/1
titanic.loc[titanic["Sex"] == 'male', "Sex"] = 0
titanic.loc[titanic["Sex"] == 'female', "Sex"] = 1

# Convert Embarked, drop S, create 2 columns C and Q
pd.crosstab(titanic["Embarked"], titanic["Survived"])
embark = pd.get_dummies(titanic["Embarked"])
embark = embark.drop(["S"], axis=1)
titanic = titanic.join(embark)
titanic = titanic.drop(["Embarked"], axis=1)

# TITLES
import re
def title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    return title.group(1)

titanic["Title"] = titanic["Name"].apply(title)
# Doing analysis on average age of each title
titles_average_age = titanic[["Age", "Title"]].groupby("Title").mean()

# Fill NaN in age with 0
titanic["Age"].fillna(value=0, inplace=True)
def filling_na_age(row):
    if row["Age"] == 0:
        row["Age"] = titles_average_age.loc[row["Title"],]
    return row
    
titanic= titanic.apply(filling_na_age, axis=1)
titanic["Age"] = titanic["Age"].astype(int)

# Check to see if there's any 0 in AGe left

table = pd.crosstab(titanic["Survived"],titanic["Title"])
# Type 1: Capt, Don, Johnkheer, Rev : 0   Mr: 81/517  Little chance of survival
# Type 2: Col, Major : 0.5 Dr: 3/7, Master: 23/40    Half chance of survival
# Type 3: Countess, Lady, Mme, Mlle, Sir, Ms : 1.0, Miss: 127/185  Mrs Highest chance of survival
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.barplot(titanic["Title"], titanic["Survived"],palette="Set3", label = "Survived rate per title")
ax.set(ylim=(0,1.1))
titles_map = {}
for k in ["Capt","Don","Jonkheer", "Rev", "Mr"]:
    titles_map[k] = 1
for k in ["Col","Major","Dr","Master"]: 
    titles_map[k] = 2
for k in ["Countess","Lady","Mme","Mlle","Sir","Ms", "Miss","Mrs"]:
    titles_map[k] = 3
    
titles = titanic["Title"]

for k,v in titles_map.items():
    titles[titles== k] = v    

# Compare titles
titles = pd.get_dummies(titles)
titles = titles.drop([1], axis=1)
titles.columns = ["Titles group 2", "Titles group 3"]
titanic = titanic.join(titles)
titanic = titanic.drop(["Title", "Name"], axis=1)

# AGE

# Graphing Age
g = sns.FacetGrid(titanic, hue="Survived", aspect = 4)
g.map(sns.kdeplot, "Age")
g.set(xlim=(0, titanic["Age"].max()))
g.add_legend()

# Survived change is high for age below 16-17
titanic.loc[titanic["Age"] < 16, "Survived"].mean()

#Survived change is high for age below 16. Thus convert 'Age' columns to 1/0 based on threshold 16
titanic.loc[titanic["Age"] <= 16,"Age"] = 1
titanic.loc[titanic["Age"] > 16, "Age"] = 0

# Pclass + Fare
sns.factorplot('Pclass', 'Survived', data=titanic, kind='bar', aspect =3, size=4)
# Pclass 1 and 2 have the highest survival rate
pclass = pd.get_dummies(titanic["Pclass"])
pclass = pclass.drop([3], axis=1)
pclass.columns = ["P1","P2"]
titanic = titanic.drop(["Pclass"], axis=1)
titanic = titanic.join(pclass)

titanic.loc[titanic["Fare"] > titanic["Fare"].mean(), "Survived"].mean()
titanic.loc[titanic["Fare"] < titanic["Fare"].mean(), "Survived"].mean()
# Higher Fare has greater chance to survive
g = sns.FacetGrid(titanic, hue="Survived", aspect = 4)
g.map(sns.kdeplot, "Fare")
g.set(xlim=(titanic["Fare"].min(), titanic["Fare"].max()))
g.add_legend()
#Fare are highly skewed with outliners 


#========================DEFINE TEST/TRAIN==============================
X = titanic.drop(["Survived"], axis=1)
y = titanic["Survived"]
from sklearn import cross_validation 

#=======================LINEAR REGRESSION===============================
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
scores = cross_validation.cross_val_score(linreg, X, y, cv = 3)
print(scores.mean())   

#=====================LOGISTIC REGRESSION===============================
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
scores = cross_validation.cross_val_score(logreg, X, y, cv = 3)
print(scores.mean())   

#=====================RANDOM FOREST=====================================0.828
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1, n_estimators = 150, min_samples_split=7,
                             min_samples_leaf=2)
scores = cross_validation.cross_val_score(rf, X, y, cv=3)
print(scores.mean())

#===================SUPPORT VECTOR MACHINE=============================0.773
from sklearn.svm import SVC
svm = SVC()
scores = cross_validation.cross_val_score(svm, X, y, cv=3)
print(scores.mean())
#===================ENSEMBLING AND GRADIENT BOOSTING=================0.829
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold

algs = [GradientBoostingClassifier(random_state=1, n_estimators = 10, max_depth=3),
        LogisticRegression(random_state=1)]

# Using cross validation folds, generate indices for train, test
kf = KFold(titanic.shape[0], random_state=1 , n_folds=3)

predictions = []
for train, test in kf:
    X_train = X.iloc[train,:]
    y_train = y.loc[train]
    X_test = X.iloc[test,:]
    combine_predict =[]
    for alg in algs:
        alg.fit(X_train, y_train)
        predict = alg.predict_proba(X_test.astype(float))[:,1]
        combine_predict.append(predict)
    
    p = (combine_predict[0]*2+combine_predict[1])/3
    p[p<=0.5] = 0
    p[p>0.5] = 1
    predictions.append(p)

predictions = np.concatenate(predictions, axis=0)

accuracy = sum(predictions[predictions ==titanic["Survived"]]) / len(predictions)
print("Accuracy: {0}".format(accuracy))


    