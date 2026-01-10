#imports

# Import Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor






#data loading and normalization

# Load datasets
train_df = pd.read_csv("../data/normalized_train_data.csv")
test_df  = pd.read_csv("../data/normalized_test_data.csv")

# Check structure
train_df.head()    

# check shape -
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)






#split of train data and test data

# Assumption:
# Last column = Target

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]


# Check Labels:
print(y_train.value_counts())