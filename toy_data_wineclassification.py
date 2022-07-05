import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

"""## Importing the dataset"""

dataset = pd.read_csv('wine.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Training the K-NN model on the Training set"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
#print(classifier.predict([[13.4,3.91,2.48,23,102,1.8,0.75,0.43,1.41,7.3,0.7,1.56,750]]))
#print(cm)
#print(accuracy_score(y_test, y_pred))
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

pickle.dump(classifier, open('toy_data_wineclassification.pkl', 'wb'))
print("Pickling Done")

model = pickle.load(open('toy_data_wineclassification.pkl', 'rb'))
print("Unpickling Done")