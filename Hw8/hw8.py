#Cris Chou
#Cyc180001
#HW8
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\C\Desktop\school\CS 4375\hw8\Auto.csv") #switch to file location of Auto.csv
print(df.head())
print(df.shape)
print('\nDescrption of mpg, weight, and year\n', df.loc[:,['mpg','weight','year']].describe())
print("The range for mpg was from 9-46.6, for weight, 1613-5140, and for year, 70-82. The averages respectively, were 23.459, 2977.584, and 76.01 ")
print("\nThe data types for each column are \n", df.dtypes)

#change using catecode
df.cylinders = df.cylinders.astype('category').cat.codes
df.origin = df.origin.astype('category')
print("\n After\n")
print(df.dtypes)

#deleting NAs
df.dropna(inplace=True)
print(df.shape)

#modify columns
averageMPG = df.mpg.mean()
df['mpg_high'] = np.where(df.mpg > averageMPG, 1,0)
df = df.drop(columns=['mpg', 'name'])
print(df.head())

#data exploration with graphs
#fig, num = plt.subplots(ncols=3)
sns.catplot(x="mpg_high", kind="count", data=df,ax=num[0])
plt.show()
#in the data there is almost an even amount of cars with high and not high mpg
sns.catplot(x="horsepower", y="weight", hue = "mpg_high",data=df,ax=num[1])
plt.show()
#in the data it seems that cars with less mpg trend towards higher horsepower and heavier weight
sns.boxplot(x = "mpg_high", y = "weight", data = df,ax=num[2])
plt.show()
#cars with lower mpg seem to average heavier weight

#train test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = df.iloc[:,0:6]
y = df.iloc[:,7]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=1234)
print("Train size", X_train.shape)
print("Test size", X_test.shape)

#Logistic regression
logreg = LogisticRegression(solver = "lbfgs")
logreg.fit(X_train,y_train)

logPred = logreg.predict(X_test)
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
print("mse= ", metrics.mean_squared_error(y_test,logPred))
print("correlation= ",metrics.r2_score(y_test,logPred))
print("Logistic Regression \n")
print(classification_report(y_test, logPred))

#Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
treePred = clf.predict(X_test)
print("Decision Tree: \n")
print(classification_report(y_test,treePred))
from sklearn import tree
tree.plot_tree(clf)
plt.show()

'''
Analysis: 
The Logistic Regression had higher accuracy for predicting when a car did not have mpg_high. However for every other metric, 
Decision Tree out performed Logistic Regression in terms of accuracy. Thus the Decision Tree performed better than Logistic Regression. 
It performed better because the target was a binary factor. There were also a few outliers which probably resulted in underfitting, since
Logistic Regression is not flexible towards outliers.
'''