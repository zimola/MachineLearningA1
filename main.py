##Matt Zimola B00475892
"""
This script follows a linear flow and isnt dont very programatically...
I was more concerned about getting the values right and having the output look nice
"""

import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import scipy.stats as stats


##=====set up data=====##
df = pd.read_csv('animals.csv', sep=",", index_col=False )
df=df.values
train=np.array(df[:,:25])
target=np.array(df[:,-1:])
target_a = [0 if target[i] =="DEER" else 1 for i in range(len(target))]
target_b = [0 if target[i] =="ELK" else 1 for i in range(len(target))]
target_c = [0 if target[i] =="CATTLE" else 1 for i in range(len(target))]
X_train_deer, X_test_deer, y_train_deer, y_test_deer=train_test_split(train, target_a, random_state=0, test_size=0.3)
X_train_elk, X_test_elk, y_train_elk, y_test_elk=train_test_split(train, target_b, random_state=0, test_size=0.3)
X_train_cattle, X_test_cattle, y_train_cattle, y_test_cattle=train_test_split(train, target_c, random_state=0, test_size=0.3)




print("-----Question 1-----")
##Question 1##
###Decision Tree
print("-----DECISION TREE-----")
print("DEER confusion Matrix and accuracy score")
clf=DecisionTreeClassifier()
clf.fit(X_train_deer,y_train_deer)
y_pred = clf.predict(X_test_deer)  ##predict my y's based on x's
print (confusion_matrix(y_test_deer,y_pred))
print("Testing Score")
print (accuracy_score(y_test_deer,y_pred)) #
y_pred=clf.predict(X_train_deer)
print("Training Score")
print(accuracy_score(y_train_deer, y_pred))

print("ELK confusion matrix and accuracy score")
clf=DecisionTreeClassifier()
clf.fit(X_train_elk,y_train_elk)
y_pred = clf.predict(X_test_elk)
print (confusion_matrix(y_test_elk,y_pred))
print("Testing Score")
print (accuracy_score(y_test_elk,y_pred))
y_pred=clf.predict(X_train_elk)
print("Training Score")
print(accuracy_score(y_train_elk, y_pred))

print("CATTLE confusion matrix and accuracy score")
clf=DecisionTreeClassifier()
clf.fit(X_train_cattle,y_train_cattle)
y_pred = clf.predict(X_test_cattle)
print (confusion_matrix(y_test_cattle,y_pred))
print("Testing Score")
print (accuracy_score(y_test_cattle,y_pred))
y_pred=clf.predict(X_train_cattle)
print("Training Score")
print(accuracy_score(y_train_cattle, y_pred))

print("-----LOGISTIC REGRESSION-----")
print("DEER confusion matrix and accuracy score")
logReg = LogisticRegression()
logReg.fit(X_train_deer, y_train_deer)
y_pred=logReg.predict(X_test_deer)
print(confusion_matrix(y_test_deer, y_pred))
print("Testing Score")
print (accuracy_score(y_test_deer,y_pred))
y_pred=logReg.predict(X_train_deer)
print("Training Score")
print(accuracy_score(y_train_deer, y_pred))

print("ELK confusion matrix and accuracy score")
logReg = LogisticRegression()
logReg.fit(X_train_elk, y_train_deer)
y_pred=logReg.predict(X_test_elk)
print(confusion_matrix(y_test_elk, y_pred))
print("Testing Score")
print (accuracy_score(y_test_elk,y_pred))
y_pred=logReg.predict(X_train_elk)
print("Training Score")
print(accuracy_score(y_train_elk, y_pred))

print("CATTLE confusion matrix and accuracy score")
logReg = LogisticRegression()
logReg.fit(X_train_cattle, y_train_cattle)
y_pred=logReg.predict(X_test_cattle)
print(confusion_matrix(y_test_cattle, y_pred))
print("Testing Score")
print (accuracy_score(y_test_cattle,y_pred))
y_pred=logReg.predict(X_train_cattle)
print("Training Score")
print(accuracy_score(y_train_cattle, y_pred))

print("-----GAUSSIAN NB-----")
print("DEER confusion matrix and accuracy score")
gnb = GaussianNB()
gnb.fit(X_train_deer, y_train_deer)
y_pred = gnb.predict(X_test_deer)
print(confusion_matrix(y_test_deer, y_pred))
print("Testing Score")
print (accuracy_score(y_test_deer,y_pred))
y_pred=gnb.predict(X_train_deer)
print("Training Score")
print(accuracy_score(y_train_deer, y_pred))

print("ELK confusion matrix and accuracy score")
gnb = GaussianNB()
gnb.fit(X_train_elk, y_train_elk)
y_pred = gnb.predict(X_test_elk)
print(confusion_matrix(y_test_elk, y_pred))
print("Testing Score")
print (accuracy_score(y_test_elk,y_pred))
y_pred=gnb.predict(X_train_elk)
print("Training Score")
print(accuracy_score(y_train_elk, y_pred))

print("CATTLE confusion matrix and accuracy score")
gnb = GaussianNB()
gnb.fit(X_train_cattle, y_train_cattle)
y_pred = gnb.predict(X_test_cattle)
print(confusion_matrix(y_test_cattle, y_pred))
print("Testing Score")
print (accuracy_score(y_test_cattle,y_pred))
y_pred=gnb.predict(X_train_cattle)
print("Training Score")
print(accuracy_score(y_train_cattle, y_pred))

print("-----RANDOM FOREST-----")
print("DEER confusion matrix and accuracy score")
rfc = RandomForestClassifier()
rfc.fit(X_train_deer, y_train_deer)
y_pred = rfc.predict(X_test_deer)
print (confusion_matrix(y_test_deer, y_pred))
print("Testing Score")
print (accuracy_score(y_test_deer,y_pred))
y_pred=rfc.predict(X_train_deer)
print("Training Score")
print(accuracy_score(y_train_deer, y_pred))

print("ELK confusion matrix and accuracy score")
rfc = RandomForestClassifier()
rfc.fit(X_train_elk, y_train_elk)
y_pred = rfc.predict(X_test_elk)
print (confusion_matrix(y_test_elk, y_pred))
print("Testing Score")
print (accuracy_score(y_test_elk,y_pred))
y_pred=rfc.predict(X_train_elk)
print("Training Score")
print(accuracy_score(y_train_elk, y_pred))

print("CATTLE confusion matrix and accuracy score")
rfc = RandomForestClassifier()
rfc.fit(X_train_cattle, y_train_cattle)
y_pred = rfc.predict(X_test_cattle)
print (confusion_matrix(y_test_cattle, y_pred))
print("Testing Score")
print (accuracy_score(y_test_cattle,y_pred))
y_pred=rfc.predict(X_train_cattle)
print("Training Score")
print(accuracy_score(y_train_cattle, y_pred))



print("                     ")
print("-----Question 2-----")
#create required classifiers
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
log = LogisticRegression()
gnb = GaussianNB()


##run each classifier though 10 fold cross validation and print mean and std
print("Random Forest 10 Fold Cross Validation")
rfcdeer = cross_val_score(rfc, train, target_a, cv=10, scoring='accuracy')
rfcelk = cross_val_score(rfc, train, target_b, cv=10, scoring='accuracy')
rfccattle = cross_val_score(rfc, train, target_c, cv=10, scoring='accuracy')
print("DEER data")
print (rfcdeer)
print("Deer: Mean and SD")
print(rfcdeer.mean(), rfcdeer.std())
print("Elk: Mean and SD")
print(rfcelk.mean(), rfcelk.std())
print("Cattle: Mean and SD")
print(rfccattle.mean(), rfccattle.std())

print("                     ")
print("Decision Tree 10 Fold Cross Validation")
dtcdeer = cross_val_score(dtc, train, target_a, cv=10, scoring='accuracy')
dtcelk = cross_val_score(dtc, train, target_b, cv=10, scoring='accuracy')
dtccattle = cross_val_score(dtc, train, target_c, cv=10, scoring='accuracy')
print("DEER data")
print (dtcdeer)
print("Deer: Mean and SD")
print(dtcdeer.mean(), dtcdeer.std())
print("Elk: Mean and SD")
print(dtcelk.mean(), dtcelk.std())
print("Cattle: Mean and SD")
print(dtccattle.mean(), dtccattle.std())
statistic, p_value = stats.ttest_rel(rfcdeer, dtcdeer)
print("stats")
print(p_value)


print("                     ")
print("Logistic Regression 10 Fold Cross Validation")
logdeer = cross_val_score(log, train, target_a, cv=10, scoring='accuracy')
logelk = cross_val_score(log, train, target_b, cv=10, scoring='accuracy')
logcattle = cross_val_score(log, train, target_c, cv=10, scoring='accuracy')
print("DEER data")
print (logdeer)
print("Deer: Mean and SD")
print(logdeer.mean(), logdeer.std())
print("Elk: Mean and SD")
print(logelk.mean(), logelk.std())
print("Cattle: Mean and SD")
print(logcattle.mean(), logcattle.std())
statistic, p_value = stats.ttest_rel(rfcdeer, logdeer)
print("stats")
print(p_value)


print("                     ")
print("GaussianNB 10 Fold Cross Validation")
gnbdeer = cross_val_score(gnb, train, target_a, cv=10, scoring='accuracy')
gnbelk = cross_val_score(gnb, train, target_b, cv=10, scoring='accuracy')
gnbcattle = cross_val_score(gnb, train, target_c, cv=10, scoring='accuracy')
print("DEER data")
print (gnbdeer)
print("Deer: Mean and SD")
print(gnbdeer.mean(), gnbdeer.std())
print("Elk: Mean and SD")
print(gnbelk.mean(), gnbelk.std())
print("Cattle: Mean and SD")
print(gnbcattle.mean(), gnbcattle.std())
statistic, p_value = stats.ttest_rel(rfcdeer, gnbdeer)
print("stats")
print(p_value)






print("                     ")
print("-----Question 3-----")


rfc10 = RandomForestClassifier(n_estimators = 10)
rfc20 = RandomForestClassifier(n_estimators = 20)
rfc50 = RandomForestClassifier(n_estimators = 50)
rfc100 = RandomForestClassifier(n_estimators = 100)

score10deer = cross_val_score(rfc10, train, target_a, cv=10, scoring='accuracy')
score20deer = cross_val_score(rfc20, train, target_a, cv=10, scoring='accuracy')
score50deer = cross_val_score(rfc50, train, target_a, cv=10, scoring='accuracy')
score100deer = cross_val_score(rfc100, train, target_a, cv=10, scoring='accuracy')

score10elk = cross_val_score(rfc10, train, target_b, cv=10, scoring='accuracy')
score20elk = cross_val_score(rfc20, train, target_b, cv=10, scoring='accuracy')
score50elk = cross_val_score(rfc50, train, target_b, cv=10, scoring='accuracy')
score100elk = cross_val_score(rfc100, train, target_b, cv=10, scoring='accuracy')

score10cattle = cross_val_score(rfc10, train, target_c, cv=10, scoring='accuracy')
score20cattle = cross_val_score(rfc20, train, target_c, cv=10, scoring='accuracy')
score50cattle = cross_val_score(rfc50, train, target_c, cv=10, scoring='accuracy')
score100cattle = cross_val_score(rfc100, train, target_c, cv=10, scoring='accuracy')

print(" ")
print("deer with 10 trees:") 
print(score10deer) 
print("mean and std") 
print(score10deer.mean())
print(score10deer.std())
print("deer with 20 trees:") 
print(score20deer) 
print("mean and std") 
print(score20deer.mean())
print(score20deer.std())
print("deer with 50 trees:") 
print(score50deer) 
print("mean and std") 
print(score50deer.mean())
print(score50deer.std())
print("deer with 100 trees:") 
print(score100deer) 
print("mean and std") 
print(score100deer.mean())
print(score100deer.std())
statistic, p_value = stats.ttest_rel(score100deer, dtcdeer)
print("stats")
print(p_value)

print("----------------")
print("Elk with 10 trees:") 
print(score10elk) 
print("mean and std") 
print(score10elk.mean())
print(score10elk.std())
print("Elk with 20 trees:") 
print(score20elk) 
print("mean and std") 
print(score20elk.mean())
print(score20elk.std())
print("Elk with 50 trees:") 
print(score50elk) 
print("mean and std") 
print(score50elk.mean())
print(score50elk.std())
print("Elk with 100 trees:") 
print(score100elk) 
print("mean and std") 
print(score100elk.mean())
print(score100elk.std())

print("----------------")
print("Cattle with 10 trees:") 
print(score10cattle) 
print("mean and std") 
print(score10cattle.mean())
print(score10cattle.std())
print("Cattle with 20 trees:") 
print(score20cattle)
print("mean and std")  
print(score20cattle.mean())
print(score20cattle.std())
print("Cattle with 50 trees:") 
print(score50cattle) 
print("mean and std") 
print(score50cattle.mean())
print(score50cattle.std())
print("Cattle with 100 trees:") 
print(score100cattle)
print("mean and std") 
print(score100cattle.mean())
print(score100cattle.std())



print("t-stats/pvalue for 100 trees")
statistic, p_value = stats.ttest_rel(score100deer, dtcdeer)
print("decision tree")
print(p_value)
statistic, p_value = stats.ttest_rel(score100deer, gnbdeer)
print("GaussianNB")
print(p_value)
statistic, p_value = stats.ttest_rel(score100deer, logdeer)
print("logreg")
print(p_value)



