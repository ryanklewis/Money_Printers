import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

window_size = 100

train_data = pd.read_csv("data/Train_NoAuction_Zscore.csv")
test_data = pd.read_csv("data/Test_NoAuction_Zscore.csv")

y_train = train_data["label_1"][window_size-1:]
y_test = test_data["label_1"][window_size-1:]

train_data = train_data.drop(
    columns=["label_1", "label_2", "label_3", "label_5", "label_10"])
test_data = test_data.drop(
    columns=["label_1", "label_2", "label_3", "label_5", "label_10"])

D = train_data.shape[1]
X_train = np.lib.stride_tricks.sliding_window_view(
    train_data, (window_size, D)).squeeze()
N, S, D = X_train.shape

X_train_flat = X_train.reshape(N, S*D)

rf = RandomForestClassifier(n_jobs=-1, verbose=True)
rf.fit(X_train_flat, y_train)

X_test = np.lib.stride_tricks.sliding_window_view(
    test_data, (window_size, D)).squeeze()
N, S, D = X_test.shape
X_test_flat = X_test.reshape(N, S*D)

y_test_predicted = rf.predict(X_test_flat)

print(classification_report(y_test, y_test_predicted))
