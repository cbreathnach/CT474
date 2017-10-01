import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# define number of neighbours here
num_neighbours = 5

# import CSV using pandas
data = pd.read_csv('illness_data.txt',sep = ',')

# convert into numpy array, X = features, y = label
X = np.array(data.drop(['test_result'],1))
y = np.array(data['test_result'])

# split the data for testing and train
# randomly splits the data so variance in results is expected
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# run the kNN for 5 neighbours and print the confusion matrix
knn = KNeighborsClassifier(n_neighbors=num_neighbours)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('K Nearest neighbour classifier on the the illness dataset')
print('Number of Neighbours:',num_neighbours)
print('Accuarcy score:',metrics.accuracy_score(y_test, y_pred))

confusion = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(confusion)