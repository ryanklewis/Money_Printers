import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

train_data = pd.read_csv("data/Train_NoAuction_Zscore.csv")
test_data = pd.read_csv("data/Test_NoAuction_Zscore.csv")

for label in ["label_1", "label_2", "label_3", "label_5", "label_10"]:
    y_train = train_data[label]
    y_test = test_data[label]

    X_train = train_data.drop(
        columns=["label_1", "label_2", "label_3", "label_5", "label_10"])
    X_test = test_data.drop(
        columns=["label_1", "label_2", "label_3", "label_5", "label_10"])

    rf = RandomForestClassifier(n_jobs=-1, verbose=True)
    rf.fit(X_train, y_train)
    y_test_predicted = rf.predict(X_test)

    print("-"*50)
    print(label)
    print(classification_report(y_test, y_test_predicted))
    print("-"*50)
