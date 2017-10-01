import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# import CSV using pandas
df = pd.read_csv('illness_data.txt',sep = ',')

# convert into numpy array, X = features, y = label
X = np.array(df.drop(['test_result'],1))
y = np.array(df['test_result'])

# range of k we want to try - test a range for kNN algorithm
k_range = range(1, 50)

# empty list to store scores
k_scores = []

# create array to store maximum value with no. neighbours
# format: [score, standard deviation, number of neighbours]
max_result = [0,0,0]

# Loop through reasonable values of k
for k in k_range:
    # run KNeighborsClassifier with k neighbours
    # add metric = 'manhattan' for manhattan distance
    # add weights = 'distance' for distance weighted
    knn = KNeighborsClassifier(n_neighbors=k)
    # obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    # append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())

    if scores.mean() > max_result[0]:
    	max_result[0] = scores.mean()
    	max_result[1] = scores.std()
    	max_result[2] = k

print('K Nearest neighbour classifier on the the illness dataset with cross validation')
print('Tested across a range of ', len(k_scores) , ' neighbours')
print('Max score:', max_result[0])
print('Standard deviation of results:', max_result[1])
print('Number of neighbours:', max_result[2])

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(x_axis, y_axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')

plt.savefig('illness_kNN.pdf')
plt.show()