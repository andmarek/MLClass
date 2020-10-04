# Copyright 2020 Forrest Sheng Bao
# GNU GPL v3.0 or later

import operator
import numpy, sklearn, sklearn.tree

def estimate_gini_impurity(feature_values, threshold, labels, polarity):
    """Compute the gini impurity for comparing a feature value against a threshold under a given polarity

    feature_values: 1D numpy array, feature_values for samples on one feature dimension
    threshold: float
    labels: 1D numpy array, the label of samples, only +1 and -1.
    polarity: operator type, only operator.gt or operator.le are allowed
Examples
    -------------
    >>> feature_values = numpy.array([1,2,3,4,5,6,7,8])
    >>> labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
    >>> for threshold in range(0,8):
    ...     estimate_gini_impurity(feature_values, threshold, labels, operator.gt)
    0.5
    0.489795918367347
    0.4444444444444444
    0.31999999999999984
    0.0
    0.0
    0.0
    0.0
    >>> for threshold in range(0,8):
    ...     estimate_gini_impurity(feature_values, threshold, labels, operator.le)
    1.0
    0.0
    0.0
    0.0
    0.0
    0.31999999999999984
    0.4444444444444445
    0.48979591836734704
    """

    # Weight average for the leaf nodes
    gini_impurity = 0

    f_c1 = feature_values[labels == +1]
    f_c2 = feature_values[labels == -1]

    v_1 = []
    v_2 = []

    for val in f_c1:
        if polarity(val, threshold):
            v_1.append(val)

    for val in f_c2:
        if polarity(val, threshold):
            v_2.append(val)

    p1 = len(v_1) / (len(v_1) + len(v_2)) # number / total in class
    p2 = len(v_2) / (len(v_1) + len(v_2)) # number / total in class

    gini_impurity = (1 - ((p2**2) + (p1**2)))

    #print("gini_impurity\n", gini_impurity)
    # YOUR CODE HERE


    return gini_impurity

def estimate_gini_impurity_expectation(feature_values, threshold, labels):
    """Compute the expectation of gini impurity given the feature values on one  feature dimension and a threshold

    feature_values: 1D numpy array, feature_values for samples on one feature dimension
    threshold: float
    labels: 1D numpy array, the label of samples, only +1 and -1.

    Examples
    ---------------
    >>> feature_values = numpy.array([1,2,3,4,5,6,7,8])
    >>> labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
    >>> for threshold in range(0,9):
    ...     estimate_gini_impurity_expectation(feature_values, threshold, labels)
    0.5
    0.4285714285714286
    0.3333333333333333
    0.1999999999999999
    0.0
    0.1999999999999999
    0.33333333333333337
    0.42857142857142866
    0.5

    """

    # YOUR CODE HERE

    return expectation

def midpoint(x):
    """Given a sequqence of numbers, return the middle points between every two consecutive ones.
    >>> x= numpy.array([1,2,3,4,5])
    >>> (x[1:] + x[:-1]) / 2
    array([1.5, 2.5, 3.5, 4.5])
    """
    return (x[1:] + x[:-1]) / 2

def grid_search_split_midpoint(X, y):
    """Given a dataset, compute the gini impurity expectation for all pairs of features and thresholds.

    Inputs
    ----------
        X: 2-D numpy array, axis 0 or row is a sample, and axis 1 or column is a feature
        y: 1-D numpy array, the labels, +1 or -1

    Returns
    ---------
        grid: 2-D numpy array, axis 0 or row is a threshold, and axis 1 or column is a feature

    Examples
    -------------
    >>> numpy.random.seed(1) # fix random number generation starting point
    >>> X = numpy.random.randint(1, 10, (8,3)) # generate training samples
    >>> y = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
    >>> grid, feature_id, bts = grid_search_split_midpoint(X, y)
    >>> print (grid)
    [[0.42857143 0.5        0.46666667]
     [0.46666667 0.5        0.46666667]
     [0.46666667 0.46666667 0.46666667]
     [0.375      0.5        0.46666667]
     [0.5        0.5        0.46666667]
     [0.5        0.5        0.5       ]
     [0.5        0.42857143 0.42857143]]
    >>> clf = sklearn.tree.DecisionTreeClassifier(max_depth=1)
    >>> clf = clf.fit(X,y)
    >>> print (clf.tree_.feature[0], clf.tree_.threshold[0], feature_id, bts)
    0 7.0 0 7.0
    >>> print(clf.tree_.feature[0] == feature_id)
    True
    >>> print( clf.tree_.threshold[0] == bts)
    True

    >>> # Antoher test case
    >>> numpy.random.seed(2) # fix random number generation starting point
    >>> X = numpy.random.randint(1, 30, (8,3)) # generate training samples
    >>> grid, feature_id, bts = grid_search_split_midpoint(X, y)
    >>> print (grid)
    [[0.42857143 0.42857143 0.42857143]
     [0.5        0.5        0.33333333]
     [0.375      0.46666667 0.46666667]
     [0.375      0.5        0.5       ]
     [0.46666667 0.46666667 0.46666667]
     [0.33333333 0.5        0.5       ]
     [0.42857143 0.42857143 0.42857143]]
    >>> clf = clf.fit(X,y) # return the sklearn DT
    >>> print (clf.tree_.feature[0], clf.tree_.threshold[0], feature_id, bts)
    2 8.5 2 8.5
    >>> print(clf.tree_.feature[0] == feature_id)
    True
    >>> print( clf.tree_.threshold[0] == bts)
    True


    >>> # yet antoher test case
    >>> numpy.random.seed(4) # fix random number generation starting point
    >>> X = numpy.random.randint(1, 100, (8,3)) # generate training samples
    >>> grid, feature_id, bts = grid_search_split_midpoint(X, y)
    >>> print (grid)
    [[0.42857143 0.42857143 0.42857143]
     [0.5        0.5        0.33333333]
     [0.46666667 0.46666667 0.375     ]
     [0.375      0.375      0.375     ]
     [0.46666667 0.2        0.46666667]
     [0.5        0.42857143 0.5       ]
     [0.42857143 0.42857143 0.42857143]]
    >>> clf = clf.fit(X,y) # return the sklearn DT
    >>> print (clf.tree_.feature[0], clf.tree_.threshold[0], feature_id, bts)
    1 47.5 1 47.5
    >>> print(clf.tree_.feature[0] == feature_id)
    True
    >>> print( clf.tree_.threshold[0] == bts)
    True
    """

    X_sorted = numpy.sort(X, axis=0)
    thresholds = numpy.apply_along_axis(midpoint, 0, X_sorted)

    # YOUR CODE HERE


    return grid, best_feature, best_threshold

def you_rock(N, R, d):
    """
    N: int, number of samples, e.g., 1000.
    R: int, maximum feature value, e.g., 100.
    d: int, number of features, e.g., 3.

    """
    numpy.random.seed() # re-random the seed
    hits = 0
    for _ in range(N):
        X = numpy.random.randint(1, R, (8,d)) # generate training samples
        y = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
        _, feature_id, bts = grid_search_split_midpoint(X, y)
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=1)
        clf = clf.fit(X,y)

        if clf.tree_.feature[0] == feature_id and clf.tree_.threshold[0] == bts:
            hits += 1
    print ("your Decision tree is {:2.2%} consistent with Scikit-learn's result.".format(hits/N))

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    #you_rock(1000, 100, 3)

    feature_values = numpy.array([1,2,3,4,5,6,7,8])
    labels = numpy.array([+1,+1,+1,+1, -1,-1,-1,-1])
    for threshold in range(0,8):
        estimate_gini_impurity(feature_values, threshold, labels, operator.gt)
