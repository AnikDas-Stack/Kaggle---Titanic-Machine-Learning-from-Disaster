# Function for assign mean value to nan in Age column
def age_assign(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age
    
# Function for checking child or not
def childkinacheck(cols):
    Age = cols[0]
    
    if Age<18:
        return 1
    else:
        return 0
    
# Function for checking alone or not
def alonekinacheck(cols):
    SibSp = cols[0]
    Parch = cols[1]
    
    if SibSp==0 and Parch==0:
        return 1
    else:
        return 0
    
# Processing Fare column where the value is nan
def fareSolve(cols):
    Fare = cols[0]
    
    if pd.isnull(Fare):
        return 43
    else:
        return Fare
    
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Importing Dataset
train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")
myTest = pd.read_csv("test.csv")

# Dropping unnecessary columns
train_dataset.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
test_dataset.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Function Calling to assign age to null values
train_dataset["Age"] = train_dataset[["Age", "Pclass"]].apply(age_assign, axis=1)
test_dataset["Age"] = test_dataset[["Age", "Pclass"]].apply(age_assign, axis=1)
test_dataset["Fare"] = test_dataset[["Fare"]].apply(fareSolve, axis=1)

# Dropping rest of the null values from train_dataset (From Embarked column)
train_dataset.dropna(inplace=True)

# LabelEncoding
train_gender = pd.get_dummies(train_dataset["Sex"])
train_embark = pd.get_dummies(train_dataset["Embarked"], drop_first=True)
test_gender = pd.get_dummies(test_dataset["Sex"])
test_embark = pd.get_dummies(test_dataset["Embarked"], drop_first=True)

# Dropping Sex and Embarked Columns
train_dataset.drop(["Sex", "Embarked"], axis=1, inplace=True)
test_dataset.drop(["Sex", "Embarked"], axis=1, inplace=True)

# Adding sex and embarked columns again 
train_dataset = pd.concat([train_dataset, train_gender, train_embark], axis=1)
test_dataset = pd.concat([test_dataset, test_gender, test_embark], axis=1)

# Checking child and alone
child_train = pd.get_dummies(train_dataset[["Age"]].apply(childkinacheck, axis=1), "Child", drop_first=True)
train_dataset = pd.concat([train_dataset, child_train], axis=1)
alone_train = pd.get_dummies(train_dataset[["SibSp", "Parch"]].apply(alonekinacheck, axis=1), "Alone", drop_first=True)
train_dataset = pd.concat([train_dataset, alone_train], axis=1)

child_test = pd.get_dummies(test_dataset[["Age"]].apply(childkinacheck, axis=1), "Child", drop_first=True)
test_dataset = pd.concat([test_dataset, child_test], axis=1)
alone_test = pd.get_dummies(test_dataset[["SibSp", "Parch"]].apply(alonekinacheck, axis=1), "Alone", drop_first=True)
test_dataset = pd.concat([test_dataset, alone_test], axis=1)

# Splitting independent & dependent variable of training dataset
X_train = train_dataset.iloc[:, train_dataset.columns != "Survived"].values
Y_train = train_dataset.iloc[:, 0].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
test_dataset = sc_X.fit_transform(test_dataset)

# Fitting SVM
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', gamma=0.1)
classifier.fit(X_train, Y_train)

# Predicting test_dataset
Y_pred_SVM = classifier.predict(test_dataset)

# Creating CSV
output = pd.DataFrame({"PassengerId": myTest.PassengerId, "Survived": Y_pred_SVM})
output.to_csv("my_submission.csv", index=False)
print("Your submission was successfully saved")










