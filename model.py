# model file for you to edit
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import train_test_split
global X_train, y_train, X, y, X_test, y_test




# Created a data processing function

def data_processing(filepath):    
    df = pd.read_csv(filepath, sep=',')
    # Shuffle the datarows randomly, to be sure that the ordering of rows is somewhat random:
    df = df.sample(frac=1)
    # Use the method "drop nan" on the DataFrame itself to force it to remove rows with null/nan-values:
    df = df.dropna(axis=0, how='any')
    limit = 12000
    print('Price limit set for classification: ${}'.format(limit))
    # Select the columns/features from the Pandas dataframe that we want to use in the model:
    X = np.array(df[['Age', 'KM']])  # we only take the first two features.
    y = 1*np.array(df['Price']>limit)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_test, y_train, y_test


# Create a linear regression model that we can train:

def model_training(X_train, y_train):
    clf = tree.DecisionTreeClassifier(max_depth=3)
    #clf.fit(X_train, y_train)
    # Train the model using CV and multiple scoring on the data we have prepared:
    cv_results = cross_validate(clf, # Provide our model to the CV-function
                            X_train, # Provide all the features (in real life only the training-data)
                            y_train, # Provide all the "correct answers" (in real life only the training-data)
                            scoring=('f1', 'precision', 'recall', 'accuracy'), 
                            cv=5 # Cross-validate using 5-fold (K-Fold method) cross-validation splits
                           )
    return cv_results


# Print some information about the linear model and its parameters:


def print_results(cv_results):
    #print(clf)
    F1_pos   = cv_results['test_f1']
    P_pos   = cv_results['test_precision']
    R_pos   = cv_results['test_recall']
    A   = cv_results['test_accuracy']
    print('\n-------------- Scores ---------------')
    print('Average F1:\t {:.2f} (+/- {:.2f})'.format(F1_pos.mean(), F1_pos.std()))
    print('Average Precision (y positive):\t {:.2f} (+/- {:.2f})'.format(P_pos.mean(), P_pos.std()))
    print('Average Recall (y positive):\t {:.2f} (+/- {:.2f})'.format(R_pos.mean(), R_pos.std()))
    print('Average Accuracy:\t {:.2f} (+/- {:.2f})'.format(A.mean(), A.std()))

file_path = "Dataset/Toyota.csv"
X_train, X_test, y_train, y_test = data_processing(file_path)
cv_results = model_training(X_train, y_train)
print_results(cv_results)

def pipeline():
    file_path = "Dataset/Toyota.csv"
    X_train, X_test, y_train, y_test = data_processing(file_path)
    cv_results = model_training(X_train, y_train)
    print_results(cv_results)