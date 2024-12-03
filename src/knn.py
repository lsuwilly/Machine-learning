import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Return the predicted class of the data according to knn, 
# after filtering by predicted label
def knn_predict(data, sample, label):

    # Filtering by cluster reduces performance
    # data = data[data["kmeans clusters"] == label]

    # Schema should be like
    # N   P   K  ...    rainfall   label  kmeans clusters

    print("WITHOUT FILTERING BY CLUSTER")
    print(data.head())

    # Separate features and target
    X = data.iloc[:, :-2].values  # All columns except the last two (clusters and classes)
    y = data.iloc[:, -2].values   # Second to last column as target (classes)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform feature scaling on the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Initialize and train the kNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)


    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Get element 0 because currently we are only passing
    # one sample case
    sample_pred = knn.predict(sample)[0]
    print("Predicted Class of Sample :",sample_pred)

    return sample_pred
