
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# define number of neighbours here - optimum of 15 from Assignment 1
num_neighbours = 15

# import CSV using pandas
data = pd.read_csv('illness_data.txt',sep = ',')

# convert into numpy array, X = features, y = label
X = np.array(data.drop(['test_result'],1))
y = np.array(data['test_result'])

# binarize the positive and negative labels for the test result
y = label_binarize(y, classes=['negative','positive'])

# split the data for testing and train
# randomly splits the data so variance in results is expected
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# run the kNN for 5 neighbours and print the confusion matrix
classifier = KNeighborsClassifier(n_neighbors=num_neighbours)
classifier.fit(X_train, y_train.ravel())

# calculate the probablities of correct prediction
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

# create the roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_prob);

# calculate the area under the curve
roc_auc_val = metrics.roc_auc_score(y_test,y_pred_prob)

# Plot the ROC curve and display the AUC
plt.figure()
lw = 1.0
plt.plot(fpr, tpr, color='red',
          lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc_val)

plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig('roc_kNN.pdf')

plt.show() 