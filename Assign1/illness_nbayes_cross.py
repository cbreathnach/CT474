import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# import CSV using pandas
df = pd.read_csv('illness_data.txt',sep = ',')

# convert into numpy array, X = features, y = label
X = np.array(df.drop(['test_result'],1))
y = np.array(df['test_result'])

# test with Naive Bayes
nb = GaussianNB()

# obtain score using 10 fold cross validation
scores = cross_val_score(nb, X, y, cv=10, scoring='accuracy')

print('Naive Bayes classifier on the the illness dataset with cross validation')
print('Accuarcy Score:',scores.mean())
print('Standard deviation of results',scores.std())