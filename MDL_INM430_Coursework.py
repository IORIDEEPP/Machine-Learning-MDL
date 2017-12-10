# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:14:34 2017

@author: Francisco J Medel
"""
# import libraries
import numpy as np # Labrary for basic mathematic equations
import matplotlib.pyplot as plt # Library to plot data
import pandas as pd # Library to manage Datasets in most efficient way
import seaborn as sns # Import seaborn library
import statsmodels.api as sm # Library to analyse stats for the models
import sklearn.preprocessing as preprocessing

from sklearn.preprocessing import Imputer # library to perform the missing values
from sklearn.cross_validation import train_test_split # slipt the data set
from sklearn.preprocessing import StandardScaler # scalar our data
from sklearn.decomposition import PCA # PCA for Highdimensionaly reduction
from sklearn.linear_model import LogisticRegression # Logistic regression model
from sklearn.ensemble import RandomForestClassifier  # RandomFores model
from sklearn.metrics import confusion_matrix # create the confusion matrix with the predicted results
from sklearn.metrics import accuracy_score # measure the accuracy of the model

# create a list column names because the file doesn't have
columnsName = ['age','workclass','fnlwgt','education',
               'education_num','marital_status','occupation',
               'relationship','race','sex','capital_gain',
               'capital_loss','hours_per_week','native_country',
               'income',]

# Import .txt file and assigne the columns name
dataset = pd.read_csv("adult_dataset.txt", names = columnsName, delimiter=",")

# validate if we have null values
dataset.isnull().sum()

# Validated that indexes have been adde to the data frame
dataset.columns

# Validate that the data has been imported correctly
dataset.head()

# count the number and columns and rows
dataset.shape

## plots the number of persons that earn more that 50>
sns.countplot(x='income',data = dataset, palette='hls')
plt.title('Distribution for Income <=50k and >50k')
plt.xlabel('Income')
plt.ylabel('Number of Records')
plt.show()

# getting the number of time an attribute is contained in the predicted variable
income_distribution = dataset['income'].value_counts()
less_50k = income_distribution[0] / dataset['income'].count()
more_50k = income_distribution[1] / dataset['income'].count()

# print the proportion of |Income for More that 50K and less that 50K
print("<=50k = %.2f" % (less_50k * 100))
print(">50K  = %.2f" % (more_50k * 100))

# convert the Predict Column Income to binary
dataset['income'] = dataset['income'].map({' <=50K': 0, ' >50K': 1})

# plot Age versus income
hist_less_50K = plt.hist(dataset[dataset.income == 1].age.values, 10, facecolor = 'red', alpha = 0.5, label = '>50K')
hist_more_50K = plt.hist(dataset[dataset.income == 0].age.values, 10, facecolor = 'orange', alpha = 0.5, label = '<=50K')
plt.title('Age vs Income 50K')
plt.xlabel('Age')
plt.ylabel('Quantity')
plt.legend()

# plot Working class vs income
f = sns.factorplot(x = "workclass",y = "income",data = dataset,kind="bar", size = 4 , palette = "Set1")
f.set_ylabels("People Percentage")
f.set_xlabels('Working Class')
plt.show()

# plot Education vs income
dis = pd.concat([dataset[dataset.income == 1].groupby('education').education.count(),
                  dataset[dataset.income == 0].groupby('education').education.count()],
                  axis = 1)
dis.columns = ['education_more_50K','education_less_50K']
dis_final = dis.education_more_50K / (dis.education_more_50K + dis.education_less_50K)
ax = dis_final.plot(kind = 'bar', color = 'red')
ax.set_xticklabels(dis_final.index, rotation = 30, fontsize = 8, ha = 'right')
ax.set_xlabel('Education')
ax.set_ylabel('People Percentage')

# plot race vs income
dis = pd.concat([dataset[dataset.income == 1].groupby('race').race.count(),
                  dataset[dataset.income == 0].groupby('race').race.count()],
                  axis = 1)
dis.columns = ['race_more_50K','race_less_50K']
#dis_final = dis.race_more_50K / (dis.race_more_50K + dis.race_less_50K)
ax = dis.plot(kind = 'bar', color = 'blue')
ax.set_xticklabels(dis_final.index, rotation = 30, fontsize = 8, ha = 'right')
ax.set_xlabel('Race')
ax.set_ylabel('People Percentage')

# plot occupation vs income
dis = pd.concat([dataset[dataset.income == 1].groupby('occupation').occupation.count(),
                  dataset[dataset.income == 0].groupby('occupation').occupation.count()],
                  axis = 1)
dis.columns = ['occupation_more_50K','occupation_less_50K']
dis_final = dis.occupation_more_50K / (dis.occupation_more_50K + dis.occupation_less_50K)
ax = dis_final.plot(kind = 'bar', color = 'red')
ax.set_xticklabels(dis_final.index, rotation = 30, fontsize = 8, ha = 'right')
ax.set_xlabel('Race')
ax.set_ylabel('People Percentage')

# plot sex vs income
dis = pd.concat([dataset[dataset.income == 1].groupby('sex').sex.count(),
                  dataset[dataset.income == 0].groupby('sex').sex.count()],
                  axis = 1)
dis.columns = ['sex_more_50K','sex_less_50K']
dis_final = dis.sex_more_50K / (dis.sex_more_50K + dis.sex_less_50K)
ax = dis_final.plot(kind = 'bar', color = 'purple')
ax.set_xticklabels(dis_final.index, rotation = 30, fontsize = 8, ha = 'right')
ax.set_xlabel('Sex')
ax.set_ylabel('People Percentage')

# plot working hours vs income
hist_less_50K = plt.hist(dataset[dataset.income == 1].hours_per_week.values, 10, facecolor = 'blue', alpha = 0.5, label = '>50K')
hist_more_50K = plt.hist(dataset[dataset.income == 0].hours_per_week.values, 10, facecolor = 'red', alpha = 0.5, label = '<=50K')
plt.title('Working Hours vs Income 50K')
plt.xlabel('Working Hours')
plt.ylabel('Quantity')
plt.legend()

# plot country of origin vs income 
dataset['native_country'].value_counts().plot(kind='bar', color = 'red')
plt.xticks(rotation = "vertical")
plt.title('Country of Origin vs Income')
plt.xlabel('Countries')
plt.ylabel('Number of People')
plt.legend()

# descriptive stats for the dataset 'capital_gain' & 'capital_loss'
dataset.describe()

# create a copy of the origianl data set without the unneeded column
encoded_dataset = dataset.copy()
encoded_dataset.drop(['fnlwgt','education_num','capital_gain','capital_loss'], axis=1, inplace=True)

# encoder all the categorical data
encoders = {}
for column in encoded_dataset.columns:
    if encoded_dataset.dtypes[column] == np.object:
        encoders[column] = preprocessing.LabelEncoder()
        encoded_dataset[column] = encoders[column].fit_transform(encoded_dataset[column])

# Calculate the correlation and plot with a heat map
g = sns.heatmap(encoded_dataset.corr(), square=True)
g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
plt.show()

# split the dataset in independent X and dependet Y variables
X = encoded_dataset.iloc[:,0:10].values
y = encoded_dataset.iloc[:, -1].values

# Taking care of missing data remplacing with the most_frequent value
imputer = Imputer(missing_values=0, strategy='most_frequent') # workclass
imputer = imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:, 1:2])

imputer = Imputer(missing_values=0, strategy='most_frequent') # occupation
imputer = imputer.fit(X[:,5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])

imputer = Imputer(missing_values=0, strategy='most_frequent') # native_country
imputer = imputer.fit(X[:,9:10])
X[:, 9:10] = imputer.transform(X[:, 9:10])

# split the data and cross_validation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80)

# Scaler the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA(n_components = None)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Visualize the PCA results
colorMappingValuesCrime = np.asarray(X_test_PCA[:,-1], 'f')
plt.suptitle('PCs for a sub group of income related variables')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.scatter(X_test_PCA[:,0], X_test_PCA[:,1],c = colorMappingValuesCrime, cmap = plt.cm.Reds, s = 50, linewidth='0')

# Fitting LogisticRegression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# accuracy Logistic Regression
acc = accuracy_score(y_test,y_pred) 
print("ACC = %.2f" % (acc))

# Predicting the Traing set
y_pred = classifier.predict(X_train)

# Making  the Confusion Matrix
cm = confusion_matrix(y_train,y_pred)

# accuracy Random Forest
acc = accuracy_score(y_train,y_pred) 
print("ACC = %.2f" % (acc))

# informing the statsmodels model
logit_sm = sm.Logit(y_train, X_train)

result = logit_sm.fit()

print (result.summary())

# Fitting Random Forest classifier to the Training set
classifier = RandomForestClassifier(n_estimators = 150, criterion = 'entropy',  random_state = 0, min_samples_split = 5)
classifier.fit(X_train, y_train)

# Predicting the Test set
y_pred = classifier.predict(X_test)

# Making  the Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

# accuracy Random Forest
acc = accuracy_score(y_test,y_pred) 
print("ACC = %.2f" % (acc))

# Predicting the Traing set
y_pred = classifier.predict(X_train)

# Making  the Confusion Matrix
cm = confusion_matrix(y_train,y_pred)

# accuracy Random Forest
acc = accuracy_score(y_train,y_pred) 
print("ACC = %.2f" % (acc))