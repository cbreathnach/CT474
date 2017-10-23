import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import KFold

import pandas as pd

## define number of neighbours here - optimum of 15 from Assignment 1
num_neighbours = 15

# import CSV using pandas
data = pd.read_csv('illness_data.txt',sep = ',')

# convert into numpy array, X = features, y = label
X = np.array(data.drop(['test_result'],1))
y = np.array(data['test_result'])

# binarize the positive and negative labels for the test result
y = label_binarize(y, classes=['negative','positive'])

#####################################################################
#               KNN
#####################################################################
knn = KNeighborsClassifier(n_neighbors=num_neighbours)

# split into k folds
kf = KFold(n_splits=10)

# arrays for the TPR and FPR values
tprs_knn = []
aucs_knn = []
base_fpr_knn = np.linspace(0, 1, 101)

plt.figure(1,figsize=(7, 7))

# loop over the folds
i = 0
for train, test in kf.split(X):
    model_knn = knn.fit(X[train], y[train].ravel())
    y_score_knn = model_knn.predict_proba(X[test])

    fpr_knn, tpr_knn, _ = roc_curve(y[test], y_score_knn[:, 1])

    roc_auc_knn = auc(fpr_knn, tpr_knn)
    aucs_knn.append(roc_auc_knn)

    plt.plot(fpr_knn, tpr_knn, alpha=0.5, lw = 0.5, label = 'ROC fold %d (AUC = %0.2f)' 
                % (i+1, roc_auc_knn))

    tpr_knn = interp(base_fpr_knn, fpr_knn, tpr_knn)
    tpr_knn[0] = 0.0
    tprs_knn.append(tpr_knn)

    i = i+1

# calculate the means and std deviations
tprs_knn = np.array(tprs_knn)
mean_tprs_knn = tprs_knn.mean(axis=0)
std_knn = tprs_knn.std(axis=0)

tprs_upper_knn = np.minimum(mean_tprs_knn + std_knn, 1)
tprs_lower_knn = mean_tprs_knn - std_knn

std_knn_auc = np.std(aucs_knn)

mean_roc_auc_knn = auc(base_fpr_knn, mean_tprs_knn)

# plot the mean curve
plt.plot(base_fpr_knn, mean_tprs_knn, 'b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' 
            % (mean_roc_auc_knn, std_knn_auc))


# plot the std deviation
plt.fill_between(base_fpr_knn, tprs_lower_knn, tprs_upper_knn, color='grey', 
                    alpha=0.3,label=r'$\pm$ 1 std. deviation')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig('roc_kNN_cross.pdf')

####################################################################
#               Naive Bayes
####################################################################

nb = GaussianNB()

# split into k folds
kf = KFold(n_splits=10)

# arrays for the TPR and FPR values
tprs_nb = []
aucs_nb = []
base_fpr_nb = np.linspace(0, 1, 101)

plt.figure(2,figsize=(7, 7))

# loop over the folds
i = 0
for train, test in kf.split(X):
    model_nb = nb.fit(X[train], y[train].ravel())
    y_score_nb = model_nb.predict_proba(X[test])

    fpr_nb, tpr_nb, _ = roc_curve(y[test], y_score_nb[:, 1])

    roc_auc_nb = auc(fpr_nb, tpr_nb)
    aucs_nb.append(roc_auc_nb)

    plt.plot(fpr_nb, tpr_nb, alpha=0.5, lw = 0.5, label = 'ROC fold %d (AUC = %0.2f)' 
                % (i+1, roc_auc_nb))

    tpr_nb = interp(base_fpr_nb, fpr_nb, tpr_nb)
    tpr_nb[0] = 0.0
    tprs_nb.append(tpr_nb)

    i = i+1

# calculate the means and std deviations
tprs_nb = np.array(tprs_nb)
mean_tprs_nb = tprs_nb.mean(axis=0)
std_nb = tprs_nb.std(axis=0)

tprs_upper_nb = np.minimum(mean_tprs_nb + std_nb, 1)
tprs_lower_nb = mean_tprs_nb - std_nb

std_nb_auc = np.std(aucs_nb)

mean_roc_auc_nb = auc(base_fpr_nb, mean_tprs_nb)

# plot the mean curve
plt.plot(base_fpr_nb, mean_tprs_nb, 'b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' 
            % (mean_roc_auc_nb, std_nb_auc))

# plot the std deviation
plt.fill_between(base_fpr_nb, tprs_lower_nb, tprs_upper_nb, color='grey', 
                    alpha=0.3,label=r'$\pm$ 1 std. deviation')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig('roc_NB_cross.pdf')

####################################################################
#               Combined Plot
####################################################################

plt.figure(3,figsize=(7, 7))

plt.plot(base_fpr_knn, mean_tprs_knn, 'darkgreen',label=r'Mean kNN ROC (AUC = %0.2f $\pm$ %0.2f)' 
            % (mean_roc_auc_knn, std_knn_auc))

plt.plot(base_fpr_nb, mean_tprs_nb, 'b',label=r'Mean Naive Bayes ROC (AUC = %0.2f $\pm$ %0.2f)' 
            % (mean_roc_auc_nb, std_nb_auc))

plt.fill_between(base_fpr_knn, tprs_lower_knn, tprs_upper_knn, color='darkgreen', 
                    alpha=0.1,label=r'$\pm$ 1 std. deviation kNN')

plt.fill_between(base_fpr_nb, tprs_lower_nb, tprs_upper_nb, color='b', 
                    alpha=0.1,label=r'$\pm$ 1 std. deviation NB')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig('roc_comb_cross.pdf')

plt.show()